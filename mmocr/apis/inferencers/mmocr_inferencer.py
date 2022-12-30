# Copyright (c) OpenMMLab. All rights reserved.
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import mmcv
import mmengine
import numpy as np
from mmengine.fileio import (get_file_backend, isdir, join_path,
                             list_dir_or_file)

from mmocr.registry import VISUALIZERS
from mmocr.structures.textdet_data_sample import TextDetDataSample
from mmocr.utils import ConfigType, bbox2poly, crop_img, poly2bbox
from .base_mmocr_inferencer import (BaseMMOCRInferencer, InputsType, PredType,
                                    ResType)
from .kie_inferencer import KIEInferencer
from .textdet_inferencer import TextDetInferencer
from .textrec_inferencer import TextRecInferencer


class MMOCRInferencer(BaseMMOCRInferencer):
    """MMOCR Inferencer. It's a wrapper around three base task
    inferenecers: TextDetInferencer, TextRecInferencer and KIEInferencer,
    and it can be used to perform end-to-end OCR or KIE inference.

    Args:
        det (Optional[Union[ConfigType, str]]): Pretrained text detection
            algorithm. It's the path to the config file or the model name
            defined in metafile. Defaults to None.
        det_weights (Optional[str]): Path to the custom checkpoint file of
            the selected det model. If it is not specified and "det" is a model
            name of metafile, the weights will be loaded from metafile.
            Defaults to None.
        rec (Optional[Union[ConfigType, str]]): Pretrained text recognition
            algorithm. It's the path to the config file or the model name
            defined in metafile. Defaults to None.
        rec_weights (Optional[str]): Path to the custom checkpoint file of
            the selected rec model. If it is not specified and "rec" is a model
            name of metafile, the weights will be loaded from metafile.
            Defaults to None.
        kie (Optional[Union[ConfigType, str]]): Pretrained key information
            extraction algorithm. It's the path to the config file or the model
            name defined in metafile. Defaults to None.
        kie_weights (Optional[str]): Path to the custom checkpoint file of
            the selected kie model. If it is not specified and "kie" is a model
            name of metafile, the weights will be loaded from metafile.
            Defaults to None.
        device (Optional[str]): Device to run inference. If None, the available
            device will be automatically used. Defaults to None.

    """

    def __init__(self,
                 det: Optional[Union[ConfigType, str]] = None,
                 det_weights: Optional[str] = None,
                 rec: Optional[Union[ConfigType, str]] = None,
                 rec_weights: Optional[str] = None,
                 kie: Optional[Union[ConfigType, str]] = None,
                 kie_weights: Optional[str] = None,
                 device: Optional[str] = None) -> None:

        if det is None and rec is None and kie is None:
            raise ValueError('At least one of det, rec and kie should be '
                             'provided.')

        self.visualizer = None
        self.num_visualized_imgs = 0

        if det is not None:
            self.textdet_inferencer = TextDetInferencer(
                det, det_weights, device)
            self.mode = 'det'
        if rec is not None:
            self.textrec_inferencer = TextRecInferencer(
                rec, rec_weights, device)
            if getattr(self, 'mode', None) == 'det':
                self.mode = 'det_rec'
                ts = str(datetime.timestamp(datetime.now()))
                self.visualizer = VISUALIZERS.build(
                    dict(
                        type='TextSpottingLocalVisualizer',
                        name=f'inferencer{ts}'))
            else:
                self.mode = 'rec'
        if kie is not None:
            if det is None or rec is None:
                raise ValueError(
                    'kie_config is only applicable when det_config and '
                    'rec_config are both provided')
            self.kie_inferencer = KIEInferencer(kie, kie_weights, device)
            self.mode = 'det_rec_kie'

    def _inputs_to_list(self, inputs: InputsType) -> list:
        """Preprocess the inputs to a list.

        Preprocess inputs to a list according to its type:

            - list or tuple: return inputs
            - str:
                - Directory path: return all files in the directory
                - normal string: return a list containing the string

        Args:
            inputs (InputsType): Inputs for the inferencer.

        Returns:
            list: List of input for the :meth:`preprocess`.
        """
        if isinstance(inputs, str):
            backend = get_file_backend(inputs)
            if hasattr(backend, 'isdir') and isdir(inputs):
                # Backends like HttpsBackend do not implement `isdir`, so only
                # those backends that implement `isdir` could accept the inputs
                # as a directory
                filename_list = list_dir_or_file(inputs, list_dir=False)
                inputs = [
                    join_path(inputs, filename) for filename in filename_list
                ]

        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        for i in range(len(inputs)):
            if not isinstance(inputs[i], np.ndarray):
                img_bytes = mmengine.fileio.get(inputs[i])
                inputs[i] = mmcv.imfrombytes(img_bytes, channel_order='rgb')

        return list(inputs)

    def forward(self, inputs: InputsType, batch_size: int,
                **forward_kwargs) -> PredType:
        """Forward the inputs to the model.

        Args:
            inputs (InputsType): The inputs to be forwarded.
            batch_size (int): Batch size. Defaults to 1.

        Returns:
            Dict: The prediction results. Possibly with keys "det", "rec", and
            "kie"..
        """
        result = {}
        if self.mode == 'rec':
            # The extra list wrapper here is for the ease of postprocessing
            self.rec_inputs = inputs
            result['rec'] = [
                self.textrec_inferencer(
                    self.rec_inputs,
                    return_datasamples=True,
                    batch_size=batch_size,
                    **forward_kwargs)
            ]
        elif self.mode.startswith('det'):
            result['det'] = self.textdet_inferencer(
                inputs,
                return_datasamples=True,
                batch_size=batch_size,
                **forward_kwargs)['predictions']
            if self.mode.startswith('det_rec'):
                result['rec'] = []
                for img, det_data_sample in zip(inputs, result['det']):
                    det_pred = det_data_sample.pred_instances
                    self.rec_inputs = []
                    for polygon in det_pred['polygons']:
                        # Roughly convert the polygon to a quadangle with
                        # 4 points
                        quad = bbox2poly(poly2bbox(polygon)).tolist()
                        self.rec_inputs.append(crop_img(img, quad))
                    result['rec'].append(
                        self.textrec_inferencer(
                            self.rec_inputs,
                            return_datasamples=True,
                            batch_size=batch_size,
                            **forward_kwargs)['predictions'])
                if self.mode == 'det_rec_kie':
                    self.kie_inputs = []
                    for img, det_data_sample, rec_data_samples in zip(
                            inputs, result['det'], result['rec']):
                        det_pred = det_data_sample.pred_instances
                        kie_input = dict(img=img)
                        kie_input['instances'] = []
                        for polygon, rec_data_sample in zip(
                                det_pred['polygons'], rec_data_samples):
                            kie_input['instances'].append(
                                dict(
                                    bbox=poly2bbox(polygon),
                                    text=rec_data_sample.pred_text.item))
                        self.kie_inputs.append(kie_input)
                    result['kie'] = self.kie_inferencer(
                        self.kie_inputs,
                        return_datasamples=True,
                        batch_size=batch_size,
                        **forward_kwargs)['predictions']
        return result

    def visualize(self, inputs: InputsType, preds: PredType,
                  **kwargs) -> List[np.ndarray]:
        """Visualize predictions.

        Args:
            inputs (List[Union[str, np.ndarray]]): Inputs for the inferencer.
            preds (List[Dict]): Predictions of the model.
            show (bool): Whether to display the image in a popup window.
                Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            draw_pred (bool): Whether to draw predicted bounding boxes.
                Defaults to True.
            pred_score_thr (float): Minimum score of bboxes to draw.
                Defaults to 0.3.
            img_out_dir (str): Output directory of images. Defaults to ''.
        """

        if 'kie' in self.mode:
            return self.kie_inferencer.visualize(self.kie_inputs, preds['kie'],
                                                 **kwargs)
        elif 'rec' in self.mode:
            if 'det' in self.mode:
                super().visualize(inputs, self._pack_e2e_datasamples(preds),
                                  **kwargs)
            else:
                return self.textrec_inferencer.visualize(
                    self.rec_inputs, preds['rec'][0], **kwargs)
        else:
            return self.textdet_inferencer.visualize(inputs, preds['det'],
                                                     **kwargs)

    def __call__(
        self,
        inputs: InputsType,
        batch_size: int = 1,
        **kwargs,
    ) -> dict:
        """Call the inferencer.

        Args:
            inputs (InputsType): Inputs for the inferencer. It can be a path
                to image / image directory, or an array, or a list of these.
            return_datasamples (bool): Whether to return results as
                :obj:`BaseDataElement`. Defaults to False.
            batch_size (int): Batch size. Defaults to 1.
            **kwargs: Key words arguments passed to :meth:`preprocess`,
                :meth:`forward`, :meth:`visualize` and :meth:`postprocess`.
                Each key in kwargs should be in the corresponding set of
                ``preprocess_kwargs``, ``forward_kwargs``, ``visualize_kwargs``
                and ``postprocess_kwargs``.

        Returns:
            dict: Inference and visualization results.
        """
        (
            preprocess_kwargs,
            forward_kwargs,
            visualize_kwargs,
            postprocess_kwargs,
        ) = self._dispatch_kwargs(**kwargs)

        ori_inputs = self.preprocess_inputs(inputs)

        preds = self.forward(ori_inputs, batch_size, **forward_kwargs)

        visualization = self.visualize(
            ori_inputs, preds,
            **visualize_kwargs)  # type: ignore  # noqa: E501
        results = self.postprocess(preds, visualization, **postprocess_kwargs)
        return results

    def postprocess(self,
                    preds: PredType,
                    visualization: Optional[List[np.ndarray]] = None,
                    print_result: bool = False,
                    pred_out_file: str = ''
                    ) -> Union[ResType, Tuple[ResType, np.ndarray]]:
        """Postprocess predictions.

        Args:
            preds (Dict): Predictions of the model.
            visualization (Optional[np.ndarray]): Visualized predictions.
            print_result (bool): Whether to print the result.
                Defaults to False.
            pred_out_file (str): Output file name to store predictions
                without images. Supported file formats are “json”, “yaml/yml”
                and “pickle/pkl”. Defaults to ''.

        Returns:
            Dict or List[Dict]: Each dict contains the inference result of
            each image. Possible keys are "det_polygons", "det_scores",
            "rec_texts", "rec_scores", "kie_labels", "kie_scores",
            "kie_edge_labels" and "kie_edge_scores".
        """

        result_dict = {}
        pred_results = [{} for _ in range(len(next(iter(preds.values()))))]
        if 'rec' in self.mode:
            for i, rec_pred in enumerate(preds['rec']):
                result = dict(rec_texts=[], rec_scores=[])
                for rec_pred_instance in rec_pred:
                    rec_dict_res = self.textrec_inferencer.pred2dict(
                        rec_pred_instance)
                    result['rec_texts'].append(rec_dict_res['text'])
                    result['rec_scores'].append(rec_dict_res['scores'])
                pred_results[i].update(result)
        if 'det' in self.mode:
            for i, det_pred in enumerate(preds['det']):
                det_dict_res = self.textdet_inferencer.pred2dict(det_pred)
                pred_results[i].update(
                    dict(
                        det_polygons=det_dict_res['polygons'],
                        det_scores=det_dict_res['scores']))
        if 'kie' in self.mode:
            for i, kie_pred in enumerate(preds['kie']):
                kie_dict_res = self.kie_inferencer.pred2dict(kie_pred)
                pred_results[i].update(
                    dict(
                        kie_labels=kie_dict_res['labels'],
                        kie_scores=kie_dict_res['scores']),
                    kie_edge_scores=kie_dict_res['edge_scores'],
                    kie_edge_labels=kie_dict_res['edge_labels'])

        result_dict['predictions'] = pred_results
        if print_result:
            print(result_dict)
        if pred_out_file != '':
            mmengine.dump(result_dict, pred_out_file)
        result_dict['visualization'] = visualization
        return result_dict

    def _pack_e2e_datasamples(self, preds: Dict) -> List[TextDetDataSample]:
        """Pack text detection and recognition results into a list of
        TextDetDataSample.

        Note that it is a temporary solution since the TextSpottingDataSample
        is not ready.
        """
        results = []
        for det_data_sample, rec_data_samples in zip(preds['det'],
                                                     preds['rec']):
            texts = []
            for rec_data_sample in rec_data_samples:
                texts.append(rec_data_sample.pred_text.item)
            det_data_sample.pred_instances.texts = texts
            results.append(det_data_sample)
        return results

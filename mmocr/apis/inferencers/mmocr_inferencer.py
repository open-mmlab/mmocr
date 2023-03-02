# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import mmcv
import mmengine
import numpy as np
from rich.progress import track

from mmocr.registry import VISUALIZERS
from mmocr.structures import TextSpottingDataSample
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
                        name=f'inferencer{ts}',
                        font_families=self.textrec_inferencer.visualizer.
                        font_families))
            else:
                self.mode = 'rec'
        if kie is not None:
            if det is None or rec is None:
                raise ValueError(
                    'kie_config is only applicable when det_config and '
                    'rec_config are both provided')
            self.kie_inferencer = KIEInferencer(kie, kie_weights, device)
            self.mode = 'det_rec_kie'

    def _inputs2ndarrray(self, inputs: List[InputsType]) -> List[np.ndarray]:
        """Preprocess the inputs to a list of numpy arrays."""
        new_inputs = []
        for item in inputs:
            if isinstance(item, np.ndarray):
                new_inputs.append(item)
            elif isinstance(item, str):
                img_bytes = mmengine.fileio.get(item)
                new_inputs.append(mmcv.imfrombytes(img_bytes))
            else:
                raise NotImplementedError(f'The input type {type(item)} is not'
                                          'supported yet.')
        return new_inputs

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
        forward_kwargs['progress_bar'] = False
        if self.mode == 'rec':
            # The extra list wrapper here is for the ease of postprocessing
            self.rec_inputs = inputs
            predictions = self.textrec_inferencer(
                self.rec_inputs,
                return_datasamples=True,
                batch_size=batch_size,
                **forward_kwargs)['predictions']
            result['rec'] = [[p] for p in predictions]
        elif self.mode.startswith('det'):  # 'det'/'det_rec'/'det_rec_kie'
            result['det'] = self.textdet_inferencer(
                inputs,
                return_datasamples=True,
                batch_size=batch_size,
                **forward_kwargs)['predictions']
            if self.mode.startswith('det_rec'):  # 'det_rec'/'det_rec_kie'
                result['rec'] = []
                for img, det_data_sample in zip(
                        self._inputs2ndarrray(inputs), result['det']):
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
                    # TODO: when the det output is empty, kie will fail
                    # as no gt-instances can be provided. It's a known
                    # issue but cannot be solved elegantly since we support
                    # batch inference.
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
                  **kwargs) -> Union[List[np.ndarray], None]:
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
            save_vis (bool): Whether to save the visualization result. Defaults
                to False.
            img_out_dir (str): Output directory of visualization results.
                If left as empty, no file will be saved. Defaults to ''.

        Returns:
            List[np.ndarray] or None: Returns visualization results only if
            applicable.
        """

        if 'kie' in self.mode:
            return self.kie_inferencer.visualize(self.kie_inputs, preds['kie'],
                                                 **kwargs)
        elif 'rec' in self.mode:
            if 'det' in self.mode:
                return super().visualize(inputs,
                                         self._pack_e2e_datasamples(preds),
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
        out_dir: str = 'results/',
        save_vis: bool = False,
        save_pred: bool = False,
        **kwargs,
    ) -> dict:
        """Call the inferencer.

        Args:
            inputs (InputsType): Inputs for the inferencer. It can be a path
                to image / image directory, or an array, or a list of these.
            batch_size (int): Batch size. Defaults to 1.
            out_dir (str): Output directory of results. Defaults to 'results/'.
            save_vis (bool): Whether to save the visualization results to
                "out_dir". Defaults to False.
            save_pred (bool): Whether to save the inference results to
                "out_dir". Defaults to False.
            **kwargs: Key words arguments passed to :meth:`preprocess`,
                :meth:`forward`, :meth:`visualize` and :meth:`postprocess`.
                Each key in kwargs should be in the corresponding set of
                ``preprocess_kwargs``, ``forward_kwargs``, ``visualize_kwargs``
                and ``postprocess_kwargs``.

        Returns:
            dict: Inference and visualization results, mapped from
                "predictions" and "visualization".
        """
        if (save_vis or save_pred) and not out_dir:
            raise ValueError('out_dir must be specified when save_vis or '
                             'save_pred is True!')
        if out_dir:
            img_out_dir = osp.join(out_dir, 'vis')
            pred_out_dir = osp.join(out_dir, 'preds')
        else:
            img_out_dir, pred_out_dir = '', ''

        (
            preprocess_kwargs,
            forward_kwargs,
            visualize_kwargs,
            postprocess_kwargs,
        ) = self._dispatch_kwargs(
            save_vis=save_vis, save_pred=save_pred, **kwargs)

        ori_inputs = self._inputs_to_list(inputs)

        chunked_inputs = super(BaseMMOCRInferencer,
                               self)._get_chunk_data(ori_inputs, batch_size)
        results = {'predictions': [], 'visualization': []}
        for ori_input in track(chunked_inputs, description='Inference'):
            preds = self.forward(ori_input, batch_size, **forward_kwargs)
            visualization = self.visualize(
                ori_input, preds, img_out_dir=img_out_dir, **visualize_kwargs)
            batch_res = self.postprocess(
                preds,
                visualization,
                pred_out_dir=pred_out_dir,
                **postprocess_kwargs)
            results['predictions'].extend(batch_res['predictions'])
            if batch_res['visualization'] is not None:
                results['visualization'].extend(batch_res['visualization'])
        return results

    def postprocess(self,
                    preds: PredType,
                    visualization: Optional[List[np.ndarray]] = None,
                    print_result: bool = False,
                    save_pred: bool = False,
                    pred_out_dir: str = ''
                    ) -> Union[ResType, Tuple[ResType, np.ndarray]]:
        """Process the predictions and visualization results from ``forward``
        and ``visualize``.

        This method should be responsible for the following tasks:

        1. Convert datasamples into a json-serializable dict if needed.
        2. Pack the predictions and visualization results and return them.
        3. Dump or log the predictions.

        Args:
            preds (PredType): Predictions of the model.
            visualization (Optional[np.ndarray]): Visualized predictions.
            print_result (bool): Whether to print the result.
                Defaults to False.
            save_pred (bool): Whether to save the inference result. Defaults to
                False.
            pred_out_dir: File to save the inference results w/o
                visualization. If left as empty, no file will be saved.
                Defaults to ''.

        Returns:
            Dict: Inference and visualization results, mapped from
                "predictions" and "visualization".
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

        if save_pred and pred_out_dir:
            pred_key = 'det' if 'det' in self.mode else 'rec'
            for pred, pred_result in zip(preds[pred_key], pred_results):
                img_path = (
                    pred.img_path if pred_key == 'det' else pred[0].img_path)
                pred_name = osp.splitext(osp.basename(img_path))[0]
                pred_name = f'{pred_name}.json'
                pred_out_file = osp.join(pred_out_dir, pred_name)
                mmengine.dump(pred_result, pred_out_file)

        result_dict['predictions'] = pred_results
        if print_result:
            print(result_dict)
        result_dict['visualization'] = visualization
        return result_dict

    def _pack_e2e_datasamples(self,
                              preds: Dict) -> List[TextSpottingDataSample]:
        """Pack text detection and recognition results into a list of
        TextSpottingDataSample."""
        results = []

        for det_data_sample, rec_data_samples in zip(preds['det'],
                                                     preds['rec']):
            texts = []
            for rec_data_sample in rec_data_samples:
                texts.append(rec_data_sample.pred_text.item)
            det_data_sample.pred_instances.texts = texts
            results.append(det_data_sample)
        return results

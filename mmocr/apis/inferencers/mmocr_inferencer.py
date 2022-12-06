# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import mmcv
import mmengine
import numpy as np

from mmocr.registry import VISUALIZERS
from mmocr.structures.textdet_data_sample import TextDetDataSample
from mmocr.utils import ConfigType, bbox2poly, crop_img, poly2bbox
from .base_mmocr_inferencer import (BaseMMOCRInferencer, InputsType, PredType,
                                    ResType)
from .kie_inferencer import KIEInferencer
from .textdet_inferencer import TextDetInferencer
from .textrec_inferencer import TextRecInferencer


class MMOCRInferencer(BaseMMOCRInferencer):

    def __init__(self,
                 det_config: Optional[Union[ConfigType, str]] = None,
                 det_ckpt: Optional[str] = None,
                 rec_config: Optional[Union[ConfigType, str]] = None,
                 rec_ckpt: Optional[str] = None,
                 kie_config: Optional[Union[ConfigType, str]] = None,
                 kie_ckpt: Optional[str] = None,
                 device: Optional[str] = None,
                 **kwargs) -> None:

        self.visualizer = None
        self.base_params = self._dispatch_kwargs(*kwargs)
        self.num_visualized_imgs = 0

        if det_config is not None:
            self.textdet_inferencer = TextDetInferencer(
                det_config, det_ckpt, device)
            self.mode = 'det'
        if rec_config is not None:
            self.textrec_inferencer = TextRecInferencer(
                rec_config, rec_ckpt, device)
            if getattr(self, 'mode', None) == 'det':
                self.mode = 'det_rec'
                ts = str(datetime.timestamp(datetime.now()))
                self.visualizer = VISUALIZERS.build(
                    dict(
                        type='TextSpottingLocalVisualizer',
                        name=f'inferencer{ts}'))
            else:
                self.mode = 'rec'
        if kie_config is not None:
            if det_config is None or rec_config is None:
                raise ValueError(
                    'kie_config is only applicable when det_config and '
                    'rec_config are both provided')
            self.kie_inferencer = KIEInferencer(kie_config, kie_ckpt, device)
            self.mode = 'det_rec_kie'

    def preprocess(self, inputs: InputsType):
        new_inputs = []
        for single_input in inputs:
            if isinstance(single_input, str):
                if osp.isdir(single_input):
                    raise ValueError('Feeding a directory is not supported')
                    # for img_path in os.listdir(single_input):
                    #     new_inputs.append(
                    #         mmcv.imread(osp.join(single_input, img_path)))
                else:
                    single_input = mmcv.imread(single_input)
                    new_inputs.append(single_input)
            else:
                new_inputs.append(single_input)
        return new_inputs

    def forward(self, inputs: InputsType) -> PredType:
        """Forward the inputs to the model.

        Args:
            inputs (InputsType): The inputs to be forwarded.
        Returns:
            Dict: The prediction results. Possibly with keys "det", "rec", and
            "kie"..
        """
        result = {}
        if self.mode == 'rec':
            # The extra list wrapper here is for the ease of postprocessing
            self.rec_inputs = inputs
            result['rec'] = [
                self.textrec_inferencer(self.rec_inputs, get_datasample=True)
            ]
        elif self.mode.startswith('det'):
            result['det'] = self.textdet_inferencer(
                inputs, get_datasample=True)
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
                            self.rec_inputs, get_datasample=True))
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
                        self.kie_inputs, get_datasample=True)
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

    def postprocess(self,
                    preds: PredType,
                    imgs: Optional[List[np.ndarray]] = None,
                    is_batch: bool = False,
                    print_result: bool = False,
                    pred_out_file: str = ''
                    ) -> Union[ResType, Tuple[ResType, np.ndarray]]:
        """Postprocess predictions.

        Args:
            preds (Dict): Predictions of the model.
            imgs (Optional[np.ndarray]): Visualized predictions.
            is_batch (bool): Whether the inputs are in a batch.
                Defaults to False.
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

        results = [{} for _ in range(len(next(iter(preds.values()))))]
        if 'rec' in self.mode:
            for i, rec_pred in enumerate(preds['rec']):
                result = dict(rec_texts=[], rec_scores=[])
                for rec_pred_instance in rec_pred:
                    rec_dict_res = self.textrec_inferencer.pred2dict(
                        rec_pred_instance)
                    result['rec_texts'].append(rec_dict_res['text'])
                    result['rec_scores'].append(rec_dict_res['scores'])
                results[i].update(result)
        if 'det' in self.mode:
            for i, det_pred in enumerate(preds['det']):
                det_dict_res = self.textdet_inferencer.pred2dict(det_pred)
                results[i].update(
                    dict(
                        det_polygons=det_dict_res['polygons'],
                        det_scores=det_dict_res['scores']))
        if 'kie' in self.mode:
            for i, kie_pred in enumerate(preds['kie']):
                kie_dict_res = self.kie_inferencer.pred2dict(kie_pred)
                results[i].update(
                    dict(
                        kie_labels=kie_dict_res['labels'],
                        kie_scores=kie_dict_res['scores']),
                    kie_edge_scores=kie_dict_res['edge_scores'],
                    kie_edge_labels=kie_dict_res['edge_labels'])

        if not is_batch:
            results = results[0]
        if print_result:
            print(results)
        if pred_out_file != '':
            mmengine.dump(results, pred_out_file)
        if imgs is None:
            return results
        return results, imgs

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

# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import mmcv
import numpy as np

from mmocr.utils import ConfigType, bbox2poly, crop_img, poly2bbox
from .base_inferencer import BaseInferencer, InputsType, PredType, ResType
from .kie_inferencer import KIEInferencer
from .textdet_inferencer import TextDetInferencer
from .textrec_inferencer import TextRecInferencer


class MMOCRInferencer(BaseInferencer):

    def __init__(self,
                 det_config: Optional[Union[ConfigType, str]] = None,
                 det_ckpt: Optional[str] = None,
                 rec_config: Optional[Union[ConfigType, str]] = None,
                 rec_ckpt: Optional[str] = None,
                 kie_config: Optional[Union[ConfigType, str]] = None,
                 kie_ckpt: Optional[str] = None,
                 device: Optional[str] = None,
                 **kwargs) -> None:

        # TODO: Use appropriate visualizer
        self.visualizer = None
        self.base_params = self.dispatch_kwargs(**kwargs)

        if det_config is not None:
            self.textdet_inferencer = TextDetInferencer(
                det_config, det_ckpt, device)
            self.mode = 'det'
        if rec_config is not None:
            self.textrec_inferencer = TextRecInferencer(
                rec_config, rec_ckpt, device)
            self.mode = 'det_rec' if getattr(self, 'mode',
                                             None) == 'det' else 'rec'
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
                single_input = mmcv.imread(single_input)
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
            result['rec'] = [self.textrec_inferencer(inputs)]
        elif self.mode.startswith('det'):
            result['det'] = self.textdet_inferencer(inputs)
            if self.mode.startswith('det_rec'):
                result['rec'] = []
                for img, det_pred in zip(inputs, result['det']):
                    cropped_imgs = []
                    for polygon in det_pred['polygons']:
                        # Roughly convert the polygon to a quadangle with
                        # 4 points
                        quad = bbox2poly(poly2bbox(polygon)).tolist()
                        cropped_imgs.append(crop_img(img, quad))
                    result['rec'].append(self.textrec_inferencer(cropped_imgs))
                if self.mode == 'det_rec_kie':
                    kie_inputs = []
                    for img, det_pred, rec_pred in zip(inputs, result['det'],
                                                       result['rec']):
                        kie_input = dict(img=img)
                        kie_input['instances'] = []
                        for polygon, poly_res in zip(det_pred['polygons'],
                                                     rec_pred):
                            kie_input['instances'].append(
                                dict(
                                    bbox=poly2bbox(polygon),
                                    text=poly_res['text']))
                        kie_inputs.append(kie_input)
                    result['kie'] = self.kie_inferencer(kie_inputs)
        return result

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
                for pred_instance in rec_pred:
                    result['rec_texts'].append(pred_instance['text'])
                    result['rec_scores'].append(pred_instance['scores'])
                results[i].update(result)
        if 'det' in self.mode:
            for i, det_pred in enumerate(preds['det']):
                results[i].update(
                    dict(
                        det_polygons=det_pred['polygons'],
                        det_scores=det_pred['scores']))
        if 'kie' in self.mode:
            for i, kie_pred in enumerate(preds['kie']):
                results[i].update(
                    dict(
                        kie_labels=kie_pred['labels'],
                        kie_scores=kie_pred['scores']),
                    kie_edge_scores=kie_pred['edge_scores'],
                    kie_edge_labels=kie_pred['edge_labels'])

        if not is_batch:
            results = results[0]
        if print_result:
            print(results)
        if pred_out_file != '':
            mmcv.dump(results, pred_out_file)
        if imgs is None:
            return results
        return results, imgs

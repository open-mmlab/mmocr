# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Sequence

import mmcv
import numpy as np
from mmengine.dataset import Compose

from mmocr.structures import KIEDataSample
from mmocr.utils import ConfigType
from .base_mmocr_inferencer import BaseMMOCRInferencer

InputType = Dict
InputsType = Sequence[Dict]


class KIEInferencer(BaseMMOCRInferencer):
    """
    Inputs:
        dict or list[dict]: A dictionary containing the following keys:
            'bbox', 'texts', ` in this format:

            - img (str or ndarray): Path to the image or the image itself.
            - img_shape (tuple(int, int)): Image shape in (H, W). In
            - instances (list[dict]): A list of instances.
              - bbox (ndarray(dtype=np.float32)): Shape (4, ). Bounding box.
              - text (str): Annotation text.

            .. code-block:: python

            {
                # A nested list of 4 numbers representing the bounding box of
                # the instance, in (x1, y1, x2, y2) order.
                'bbox': np.array([[x1, y1, x2, y2], [x1, y1, x2, y2], ...],
                                dtype=np.int32),

                # List of texts.
                "texts": ['text1', 'text2', ...],
            }
    """

    def _init_pipeline(self, cfg: ConfigType) -> None:
        """Initialize the test pipeline."""
        pipeline_cfg = cfg.test_dataloader.dataset.pipeline
        idx = self._get_transform_idx(pipeline_cfg, 'LoadKIEAnnotations')
        if idx == -1:
            raise ValueError(
                'LoadKIEAnnotations is not found in the test pipeline')
        pipeline_cfg[idx]['with_label'] = False
        self.novisual = self._get_transform_idx(pipeline_cfg,
                                                'LoadImageFromFile') == -1
        # If it's in non-visual mode, self.pipeline will be specified.
        # Otherwise, file_pipeline and ndarray_pipeline will be specified.
        if self.novisual:
            self.pipeline = Compose(pipeline_cfg)
        else:
            return super()._init_pipeline(cfg)

    def preprocess(self, inputs: InputsType) -> List[Dict]:
        results = []
        for single_input in inputs:
            if self.novisual:
                if 'img' not in single_input and \
                  'img_shape' not in single_input:
                    raise ValueError(
                        'KIEInferencer in no-visual mode '
                        'requires input has "img" or "img_shape", but both are'
                        ' not found.')
                if 'img' in single_input:
                    new_input = {
                        k: v
                        for k, v in single_input.items() if k != 'img'
                    }
                    img = single_input['img']
                    if isinstance(img, str):
                        img = mmcv.imread(img)
                    new_input['img_shape'] = img.shape[::2]
                results.append(self.pipeline(new_input))
            else:
                if 'img' not in single_input:
                    raise ValueError(
                        'This inferencer is constructed to '
                        'accept image inputs, but the input does not contain '
                        '"img" key.')
                if isinstance(single_input['img'], str):
                    data_ = {
                        k: v
                        for k, v in single_input.items() if k != 'img'
                    }
                    data_['img_path'] = single_input['img']
                    results.append(self.file_pipeline(data_))
                elif isinstance(single_input['img'], np.ndarray):
                    results.append(self.ndarray_pipeline(single_input))
                else:
                    atype = type(single_input['img'])
                    raise ValueError(f'Unsupported input type: {atype}')
        return results

    def _pred2dict(self, data_sample: KIEDataSample) -> Dict:
        """Extract elements necessary to represent a prediction into a
        dictionary. It's better to contain only basic data elements such as
        strings and numbers in order to guarantee it's json-serializable.

        Args:
            data_sample (TextRecogDataSample): The data sample to be converted.

        Returns:
            dict: The output dictionary.
        """
        result = {}
        pred = data_sample.pred_instances
        result['scores'] = pred.scores.cpu().numpy().tolist()
        result['edge_scores'] = pred.edge_scores.cpu().numpy().tolist()
        result['edge_labels'] = pred.edge_labels.cpu().numpy().tolist()
        result['labels'] = pred.labels.cpu().numpy().tolist()
        return result

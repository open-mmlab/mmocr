# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import numpy as np
from mmengine.dataset import Compose

from mmocr.structures import TextRecogDataSample
from mmocr.utils import ConfigType
from .base_inferencer import BaseInferencer


class TextRecInferencer(BaseInferencer):

    def _init_pipeline(self, cfg: ConfigType) -> None:
        """Initialize the test pipeline.

        TODO: To be removed after multi-evaluator has been implemented.
        """
        pipeline_cfg = cfg.test_pipeline

        # For inference, the key of ``instances`` is not used.
        pipeline_cfg[-1]['meta_keys'] = tuple(
            meta_key for meta_key in pipeline_cfg[-1]['meta_keys']
            if meta_key != 'instances')

        # Loading annotations is also not applicable
        idx = self._get_transform_idx(pipeline_cfg, 'LoadOCRAnnotations')
        if idx != -1:
            del pipeline_cfg[idx]

        self.file_pipeline = Compose(pipeline_cfg)

        load_img_idx = self._get_transform_idx(pipeline_cfg,
                                               'LoadImageFromFile')
        if load_img_idx == -1:
            raise ValueError(
                'LoadImageFromFile is not found in the test pipeline')
        pipeline_cfg[load_img_idx] = dict(type='mmdet.LoadImageFromNDArray')

        self.ndarray_pipeline = Compose(pipeline_cfg)

    def _pred2dict(self, data_sample: TextRecogDataSample) -> Dict:
        """Extract elements necessary to represent a prediction into a
        dictionary. It's better to contain only basic data elements such as
        strings and numbers in order to guarantee it's json-serializable.

        Args:
            data_sample (TextRecogDataSample): The data sample to be converted.

        Returns:
            dict: The output dictionary.
        """
        result = {}
        result['text'] = data_sample.pred_text.item
        result['scores'] = float(np.mean(data_sample.pred_text.score))
        return result

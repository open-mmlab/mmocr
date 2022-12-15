# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import torch

from mmocr.models.textdet.detectors.base import BaseTextDetector
from mmocr.registry import MODELS
from mmocr.utils import OptConfigType, OptDetSampleList, OptMultiConfig


@MODELS.register_module()
class TwoStageTextSpotter(BaseTextDetector):
    """Two-stage text spotter.

    Args:
        backbone (dict, optional): Config dict for text spotter backbone.
            Defaults to None.
        neck (dict, optional): Config dict for text spotter neck. Defaults to
            None.
        det_head (dict, optional): Config dict for text spotter head. Defaults
            to None.
        roi_head (dict, optional): Config dict for text spotter roi head.
            Defaults to None.
        data_preprocessor (dict, optional): Config dict for text spotter data
            preprocessor. Defaults to None.
        init_cfg (dict, optional): Initialization config dict. Defaults to
            None.
    """

    def __init__(self,
                 backbone: OptConfigType = None,
                 neck: OptConfigType = None,
                 det_head: OptConfigType = None,
                 roi_head: OptConfigType = None,
                 postprocessor: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:

        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        if det_head is not None:
            self.det_head = MODELS.build(det_head)

        if roi_head is not None:
            self.roi_head = MODELS.build(roi_head)

        if postprocessor is not None:
            self.postprocessor = MODELS.build(postprocessor)

    @property
    def with_det_head(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'det_head') and self.det_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def loss(self, inputs: torch.Tensor,
             data_samples: OptDetSampleList) -> Dict:
        pass

    def predict(self, inputs: torch.Tensor,
                data_samples: OptDetSampleList) -> OptDetSampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing."""
        inputs = self.extract_feat(inputs)
        data_samples = self.det_head.predict(inputs, data_samples)
        rec_data_samples = self.roi_head.predict(inputs, data_samples)
        return self.postprocessor(data_samples, rec_data_samples)

    def _forward(self, inputs: torch.Tensor,
                 data_samples: OptDetSampleList) -> torch.Tensor:
        pass

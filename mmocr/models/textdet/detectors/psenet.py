# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.models.builder import DETECTORS
from .base_text_detector import BaseTextDetector
from .single_stage_text_detector import SingleStageTextDetector


@DETECTORS.register_module()
class PSENet(BaseTextDetector, SingleStageTextDetector):
    """The class for implementing PSENet text detector: Shape Robust Text
    Detection with Progressive Scale Expansion Network.

    [https://arxiv.org/abs/1806.02559].
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 show_score=False,
                 init_cfg=None):
        SingleStageTextDetector.__init__(self, backbone, neck, bbox_head,
                                         train_cfg, test_cfg, pretrained,
                                         init_cfg)
        BaseTextDetector.__init__(self, show_score)

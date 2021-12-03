# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.models.builder import DETECTORS
from .base_text_detector import BaseTextDetector
from .single_stage_text_detector import SingleStageTextDetector


@DETECTORS.register_module()
class TextSnake(BaseTextDetector, SingleStageTextDetector):
    """The class for implementing TextSnake text detector: TextSnake: A
    Flexible Representation for Detecting Text of Arbitrary Shapes.

    [https://arxiv.org/abs/1807.01544]
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

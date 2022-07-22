# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.registry import MODELS
from .single_stage_text_detector import SingleStageTextDetector


@MODELS.register_module()
class PANet(SingleStageTextDetector):
    """The class for implementing PANet text detector:

    Efficient and Accurate Arbitrary-Shaped Text Detection with Pixel
    Aggregation Network [https://arxiv.org/abs/1908.05900].
    """

# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.registry import MODELS
from .single_stage_text_detector import SingleStageTextDetector


@MODELS.register_module()
class PSENet(SingleStageTextDetector):
    """The class for implementing PSENet text detector: Shape Robust Text
    Detection with Progressive Scale Expansion Network.

    [https://arxiv.org/abs/1806.02559].
    """

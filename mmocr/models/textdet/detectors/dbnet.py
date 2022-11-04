# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.registry import MODELS
from .single_stage_text_detector import SingleStageTextDetector


@MODELS.register_module()
class DBNet(SingleStageTextDetector):
    """The class for implementing DBNet text detector: Real-time Scene Text
    Detection with Differentiable Binarization.

    [https://arxiv.org/abs/1911.08947].
    """

# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.registry import MODELS
from .single_stage_text_detector import SingleStageTextDetector


@MODELS.register_module()
class TextSnake(SingleStageTextDetector):
    """The class for implementing TextSnake text detector: TextSnake: A
    Flexible Representation for Detecting Text of Arbitrary Shapes.

    [https://arxiv.org/abs/1807.01544]
    """

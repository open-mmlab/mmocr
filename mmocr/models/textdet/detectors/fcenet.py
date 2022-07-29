# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.registry import MODELS
from .single_stage_text_detector import SingleStageTextDetector


@MODELS.register_module()
class FCENet(SingleStageTextDetector):
    """The class for implementing FCENet text detector
    FCENet(CVPR2021): Fourier Contour Embedding for Arbitrary-shaped Text
        Detection

    [https://arxiv.org/abs/2104.10442]
    """

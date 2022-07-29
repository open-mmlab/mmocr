# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.registry import MODELS
from .single_stage_text_detector import SingleStageTextDetector


@MODELS.register_module()
class DRRG(SingleStageTextDetector):
    """The class for implementing DRRG text detector. Deep Relational Reasoning
    Graph Network for Arbitrary Shape Text Detection.

    [https://arxiv.org/abs/2003.07493]
    """

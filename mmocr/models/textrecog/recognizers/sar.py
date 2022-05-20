# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.registry import MODELS
from .encode_decode_recognizer import EncodeDecodeRecognizer


@MODELS.register_module()
class SARNet(EncodeDecodeRecognizer):
    """Implementation of `SAR <https://arxiv.org/abs/1811.00751>`_"""

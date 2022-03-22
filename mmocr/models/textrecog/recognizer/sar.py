# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.models.builder import RECOGNIZERS
from .encode_decode_recognizer import EncodeDecodeRecognizer


@RECOGNIZERS.register_module()
class SARNet(EncodeDecodeRecognizer):
    """Implementation of `SAR <https://arxiv.org/abs/1811.00751>`_"""

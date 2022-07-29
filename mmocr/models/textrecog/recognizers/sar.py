# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.registry import MODELS
from .encoder_decoder_recognizer import EncoderDecoderRecognizer


@MODELS.register_module()
class SARNet(EncoderDecoderRecognizer):
    """Implementation of `SAR <https://arxiv.org/abs/1811.00751>`_"""

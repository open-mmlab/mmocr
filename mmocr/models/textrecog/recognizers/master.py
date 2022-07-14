# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.registry import MODELS
from .encoder_decoder_recognizer import EncoderDecoderRecognizer


@MODELS.register_module()
class MASTER(EncoderDecoderRecognizer):
    """Implementation of `MASTER <https://arxiv.org/abs/1910.02562>`_"""

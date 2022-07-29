# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.registry import MODELS
from .encoder_decoder_recognizer import EncoderDecoderRecognizer


@MODELS.register_module()
class SATRN(EncoderDecoderRecognizer):
    """Implementation of `SATRN <https://arxiv.org/abs/1910.04396>`_"""

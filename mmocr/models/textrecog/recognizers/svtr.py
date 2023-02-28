# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.registry import MODELS
from .encoder_decoder_recognizer import EncoderDecoderRecognizer


@MODELS.register_module()
class SVTR(EncoderDecoderRecognizer):
    """A PyTorch implementation of : `SVTR: Scene Text Recognition with a
    Single Visual Model <https://arxiv.org/abs/2205.00159>`_"""

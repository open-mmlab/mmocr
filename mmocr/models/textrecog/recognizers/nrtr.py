# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.registry import MODELS
from .encoder_decoder_recognizer import EncoderDecoderRecognizer


@MODELS.register_module()
class NRTR(EncoderDecoderRecognizer):
    """Implementation of `NRTR <https://arxiv.org/pdf/1806.00926.pdf>`_"""

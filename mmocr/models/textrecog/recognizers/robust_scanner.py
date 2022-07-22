# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.registry import MODELS
from .encoder_decoder_recognizer import EncoderDecoderRecognizer


@MODELS.register_module()
class RobustScanner(EncoderDecoderRecognizer):
    """Implementation of `RobustScanner.

    <https://arxiv.org/pdf/2007.07542.pdf>
    """

# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.registry import MODELS
from .encoder_decoder_recognizer import EncoderDecoderRecognizer


@MODELS.register_module()
class ASTER(EncoderDecoderRecognizer):
    """Implement  `ASTER: An Attentional Scene Text Recognizer with Flexible
    Rectification.

    <https://ieeexplore.ieee.org/abstract/document/8395027/`
    """

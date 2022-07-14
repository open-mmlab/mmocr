# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.registry import MODELS
from .encoder_decoder_recognizer import EncoderDecoderRecognizer


@MODELS.register_module()
class CRNNNet(EncoderDecoderRecognizer):
    """CTC-loss based recognizer."""

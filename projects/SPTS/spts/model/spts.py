# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.registry import MODELS
from .encoder_decoder_text_spotter import EncoderDecoderTextSpotter


@MODELS.register_module()
class SPTS(EncoderDecoderTextSpotter):
    """SPTS."""

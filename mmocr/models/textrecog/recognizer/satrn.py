# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.models.builder import RECOGNIZERS
from .encode_decode_recognizer import EncodeDecodeRecognizer


@RECOGNIZERS.register_module()
class SATRN(EncodeDecodeRecognizer):
    """Implementation of `SATRN <https://arxiv.org/abs/1910.04396>`_"""

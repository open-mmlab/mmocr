# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.registry import MODELS
from .encode_decode_recognizer import EncodeDecodeRecognizer


@MODELS.register_module()
class SATRN(EncodeDecodeRecognizer):
    """Implementation of `SATRN <https://arxiv.org/abs/1910.04396>`_"""

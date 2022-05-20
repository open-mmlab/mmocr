# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.registry import MODELS
from .encode_decode_recognizer import EncodeDecodeRecognizer


@MODELS.register_module()
class MASTER(EncodeDecodeRecognizer):
    """Implementation of `MASTER <https://arxiv.org/abs/1910.02562>`_"""

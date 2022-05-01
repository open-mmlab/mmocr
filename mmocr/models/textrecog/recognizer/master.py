# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.models.builder import DETECTORS
from .encode_decode_recognizer import EncodeDecodeRecognizer


@DETECTORS.register_module()
class MASTER(EncodeDecodeRecognizer):
    """Implementation of `MASTER <https://arxiv.org/abs/1910.02562>`_"""

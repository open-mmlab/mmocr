# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.registry import MODELS
from .encode_decode_recognizer import EncodeDecodeRecognizer


@MODELS.register_module()
class NRTR(EncodeDecodeRecognizer):
    """Implementation of `NRTR <https://arxiv.org/pdf/1806.00926.pdf>`_"""

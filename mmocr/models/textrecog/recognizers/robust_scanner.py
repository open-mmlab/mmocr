# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.registry import MODELS
from .encode_decode_recognizer import EncodeDecodeRecognizer


@MODELS.register_module()
class RobustScanner(EncodeDecodeRecognizer):
    """Implementation of `RobustScanner.

    <https://arxiv.org/pdf/2007.07542.pdf>
    """

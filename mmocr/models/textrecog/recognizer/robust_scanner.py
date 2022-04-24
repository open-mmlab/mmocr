# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.models.builder import RECOGNIZERS
from .encode_decode_recognizer import EncodeDecodeRecognizer


@RECOGNIZERS.register_module()
class RobustScanner(EncodeDecodeRecognizer):
    """Implementation of `RobustScanner.

    <https://arxiv.org/pdf/2007.07542.pdf>
    """

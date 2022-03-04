# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.models.builder import RECOGNIZERS
from .encode_decode_recognizer import EncodeDecodeRecognizer


@RECOGNIZERS.register_module()
class CRNNNet(EncodeDecodeRecognizer):
    """CTC-loss based recognizer."""

# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.registry import MODELS
from .encode_decode_recognizer import EncodeDecodeRecognizer


@MODELS.register_module()
class CRNNNet(EncodeDecodeRecognizer):
    """CTC-loss based recognizer."""

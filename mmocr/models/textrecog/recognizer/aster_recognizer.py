# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.models.builder import DETECTORS
from .encode_decode_recognizer import EncodeDecodeRecognizer


@DETECTORS.register_module()
class ASTERNet(EncodeDecodeRecognizer):
    """CELoss based recognizer."""

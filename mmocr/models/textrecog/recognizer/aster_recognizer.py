# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from mmocr.models.builder import (DETECTORS, build_backbone, build_convertor,
                                  build_decoder, build_encoder, build_fuser,
                                  build_loss, build_preprocessor)
from .encode_decode_recognizer import EncodeDecodeRecognizer


@DETECTORS.register_module()
class ASTERNet(EncodeDecodeRecognizer):
    """CELoss based recognizer."""

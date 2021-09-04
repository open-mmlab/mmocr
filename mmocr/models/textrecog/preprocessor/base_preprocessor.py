# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import BaseModule

from mmocr.models.builder import PREPROCESSOR


@PREPROCESSOR.register_module()
class BasePreprocessor(BaseModule):
    """Base Preprocessor class for text recognition."""

    def forward(self, x, **kwargs):
        return x

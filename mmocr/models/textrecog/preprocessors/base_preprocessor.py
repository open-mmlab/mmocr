# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import BaseModule

from mmocr.registry import MODELS


@MODELS.register_module()
class BasePreprocessor(BaseModule):
    """Base Preprocessor class for text recognition."""

    def forward(self, x, **kwargs):
        return x

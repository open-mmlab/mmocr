# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model import BaseModule

from mmocr.registry import MODELS


@MODELS.register_module()
class BasePreprocessor(BaseModule):
    """Base Preprocessor class for text recognition."""

    def forward(self, x, **kwargs):
        return x

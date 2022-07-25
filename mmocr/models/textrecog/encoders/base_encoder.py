# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model import BaseModule

from mmocr.registry import MODELS


@MODELS.register_module()
class BaseEncoder(BaseModule):
    """Base Encoder class for text recognition."""

    def forward(self, feat, **kwargs):
        return feat

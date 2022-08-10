# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch.nn as nn
from mmdet.models.backbones import MobileNetV2 as MMDet_MobileNetV2
from torch import Tensor

from mmocr.registry import MODELS
from mmocr.utils.typing import InitConfigType


@MODELS.register_module()
class MobileNetV2(MMDet_MobileNetV2):
    """See mmdet.models.backbones.MobileNetV2 for details.

    Args:
        pooling_layers (list): List of indices of pooling layers.
        init_cfg (InitConfigType, optional): Initialization config dict.
    """
    # Parameters to build layers. 4 parameters are needed to construct a
    # layer, from left to right: expand_ratio, channel, num_blocks, stride.
    arch_settings = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 1],
                     [6, 64, 4, 1], [6, 96, 3, 1], [6, 160, 3, 1],
                     [6, 320, 1, 1]]

    def __init__(self,
                 pooling_layers: List = [3, 4, 5],
                 init_cfg: InitConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.pooling = nn.MaxPool2d((2, 2), (2, 1), (0, 1))
        self.pooling_layers = pooling_layers

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""

        x = self.conv1(x)
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.pooling_layers:
                x = self.pooling(x)

        return x

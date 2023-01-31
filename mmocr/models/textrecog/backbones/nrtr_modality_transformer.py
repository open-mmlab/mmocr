# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn

from mmengine.model import BaseModule
from mmocr.registry import MODELS


@MODELS.register_module()
class NRTRModalityTransform(BaseModule):
    """Modality transform in NRTR.

    Args:
        in_channels (int): Input channel of image. Defaults to 3.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(
        self,
        in_channels: int = 3,
        init_cfg: Optional[Union[Dict, Sequence[Dict]]] = [
            dict(type='Kaiming', layer='Conv2d'),
            dict(type='Uniform', layer='BatchNorm2d')
        ]
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1)
        self.relu_1 = nn.ReLU(True)
        self.bn_1 = nn.BatchNorm2d(32)

        self.conv_2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1)
        self.relu_2 = nn.ReLU(True)
        self.bn_2 = nn.BatchNorm2d(64)

        self.linear = nn.Linear(512, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone forward.

        Args:
            x (torch.Tensor): Image tensor of shape :math:`(N, C, W, H)`. W, H
                is the width and height of image.
        Return:
            Tensor: Output tensor.
        """
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.bn_1(x)

        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.bn_2(x)

        n, c, h, w = x.size()

        x = x.permute(0, 3, 2, 1).contiguous().view(n, w, h * c)

        x = self.linear(x)

        x = x.permute(0, 2, 1).contiguous().view(n, -1, 1, w)

        return x

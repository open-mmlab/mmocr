# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmocr.registry import MODELS


@MODELS.register_module()
class ShallowCNN(BaseModule):
    """Implement Shallow CNN block for SATRN.

    SATRN: `On Recognizing Texts of Arbitrary Shapes with 2D Self-Attention
    <https://arxiv.org/pdf/1910.04396.pdf>`_.

    Args:
        input_channels (int): Number of channels of input image tensor
            :math:`D_i`. Defaults to 1.
        hidden_dim (int): Size of hidden layers of the model :math:`D_m`.
            Defaults to 512.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(
        self,
        input_channels: int = 1,
        hidden_dim: int = 512,
        init_cfg: Optional[Union[Dict, List[Dict]]] = [
            dict(type='Kaiming', layer='Conv2d'),
            dict(type='Uniform', layer='BatchNorm2d')
        ]
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(input_channels, int)
        assert isinstance(hidden_dim, int)

        self.conv1 = ConvModule(
            input_channels,
            hidden_dim // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))
        self.conv2 = ConvModule(
            hidden_dim // 2,
            hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input image feature :math:`(N, D_i, H, W)`.

        Returns:
            Tensor: A tensor of shape :math:`(N, D_m, H/4, W/4)`.
        """

        x = self.conv1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.pool(x)

        return x

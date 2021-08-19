# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from mmdet.models.builder import BACKBONES


@BACKBONES.register_module()
class ShallowCNN(BaseModule):
    """Implement Shallow CNN block for SATRN, see
      `SATRN <https://arxiv.org/pdf/1910.04396.pdf>`_
    Args:
        base_channels (int): Number of channels of input image tensor.
        hidden_dim (int): Size of hidden layers of the model.
    """

    def __init__(self,
                 input_channels=1,
                 hidden_dim=512,
                 init_cfg=[
                     dict(type='Kaiming', layer='Conv2d'),
                     dict(type='Uniform', layer='BatchNorm2d')
                 ]):
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

    def forward(self, x):

        x = self.conv1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.pool(x)

        return x

# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule, Sequential

from mmocr.registry import MODELS


@MODELS.register_module()
class ABCNetRecBackbone(BaseModule):

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)

        self.convs = Sequential(
            ConvModule(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1,
                bias='auto',
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1,
                bias='auto',
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1,
                stride=(2, 1),
                bias='auto',
                norm_cfg=dict(type='GN', num_groups=32),
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1,
                stride=(2, 1),
                bias='auto',
                norm_cfg=dict(type='GN', num_groups=32),
                act_cfg=dict(type='ReLU')),
            nn.AvgPool2d(kernel_size=(2, 1), stride=1))

    def forward(self, x):
        return self.convs(x)

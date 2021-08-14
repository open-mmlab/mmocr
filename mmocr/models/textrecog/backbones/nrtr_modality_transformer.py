import torch.nn as nn
from mmcv.runner import BaseModule

from mmocr.models.builder import BACKBONES


@BACKBONES.register_module()
class NRTRModalityTransform(BaseModule):

    def __init__(self,
                 input_channels=3,
                 input_height=32,
                 init_cfg=[
                     dict(type='Kaiming', layer='Conv2d'),
                     dict(type='Uniform', layer='BatchNorm2d')
                 ]):
        super().__init__(init_cfg=init_cfg)

        self.conv_1 = nn.Conv2d(
            in_channels=input_channels,
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

        feat_height = input_height // 4

        self.linear = nn.Linear(64 * feat_height, 512)

    def forward(self, x):
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

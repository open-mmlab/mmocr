# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn.resnet import BasicBlock as MMCV_BasicBlock
from mmcv.cnn.resnet import conv3x3


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(MMCV_BasicBlock):

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 use_conv1x1=False,
                 style='pytorch',
                 with_cp=False):
        super().__init__(
            inplanes,
            planes,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            style=style,
            with_cp=with_cp)
        if use_conv1x1:
            self.conv1 = conv1x1(inplanes, planes)
            self.conv2 = conv3x3(planes, planes, stride)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes, planes * self.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out

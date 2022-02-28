# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner import BaseModule

from mmocr.models.builder import BACKBONES
from mmocr.models.textrecog.layers import ContextBlock


def conv3x3(in_planes, out_planes, stride=1):
    """ 3x3 convolution with padding """
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """ 1x1 convolution """
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(
                    self,
                    inplanes,
                    planes,
                    stride=1,
                    downsample=None,
                    gcb_config=None
    ):
        super(_BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.9)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.9)
        self.downsample = downsample
        self.stride = stride
        self.gcb_config = gcb_config

        if self.gcb_config is not None:
            gcb_ratio = gcb_config['ratio']
            gcb_headers = gcb_config['headers']
            att_scale = gcb_config['att_scale']
            fusion_type = gcb_config['fusion_type']
            self.context_block = ContextBlock(
                                                inplanes=planes,
                                                ratio=gcb_ratio,
                                                headers=gcb_headers,
                                                att_scale=att_scale,
                                                fusion_type=fusion_type
                                            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.gcb_config is not None:
            out = self.context_block(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def get_gcb_config(gcb_config, layer):
    if gcb_config is None or not gcb_config['layers'][layer]:
        return None
    else:
        return gcb_config


@BACKBONES.register_module()
class ResNetMASTER(BaseModule):
    """Backbone module in `MASTER.

    <https://arxiv.org/abs/1910.02562>`_.

    Args:
        layers (list): Block number of each stage.
        input_dim (int): Channel number of image.
        gcb_config (dict): Global context block setting dict.
    """

    def __init__(self, layers, input_dim=3, gcb_config=None):
        assert len(layers) >= 4

        super(ResNetMASTER, self).__init__()
        self.inplanes = 128
        self.conv1 = nn.Conv2d(input_dim, 64,
                               kernel_size=3,
                               stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 128,
                               kernel_size=3,
                               stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer1 = self._make_layer(
                                    _BasicBlock, 256,
                                    layers[0], stride=1,
                                    gcb_config=get_gcb_config(gcb_config, 0)
        )

        self.conv3 = nn.Conv2d(256, 256,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer2 = self._make_layer(
                                    _BasicBlock, 256,
                                    layers[1], stride=1,
                                    gcb_config=get_gcb_config(gcb_config, 1)
        )

        self.conv4 = nn.Conv2d(256, 256,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)

        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.layer3 = self._make_layer(
                                    _BasicBlock, 512,
                                    layers[2], stride=1,
                                    gcb_config=get_gcb_config(gcb_config, 2)
        )

        self.conv5 = nn.Conv2d(512, 512,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU(inplace=True)

        self.layer4 = self._make_layer(
                                    _BasicBlock, 512, layers[3], stride=1,
                                    gcb_config=get_gcb_config(gcb_config, 3)
        )

        self.conv6 = nn.Conv2d(512, 512,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU(inplace=True)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu'
                                        )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, gcb_config=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
                        block(
                                self.inplanes,
                                planes,
                                stride,
                                downsample,
                                gcb_config=gcb_config
                        )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        f = []
        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        # (48, 160)

        x = self.maxpool1(x)
        x = self.layer1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        f.append(x)
        # (24, 80)

        x = self.maxpool2(x)
        x = self.layer2(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        f.append(x)
        # (12, 40)

        x = self.maxpool3(x)

        x = self.layer3(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.layer4(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        f.append(x)
        # (6, 40)

        return f

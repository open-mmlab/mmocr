# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner import BaseModule, Sequential

import mmocr.utils as utils
from mmocr.models.builder import BACKBONES
from mmocr.models.textrecog.layers import BasicBlock


@BACKBONES.register_module()
class ResNetABI(BaseModule):
    """Implement ResNet backbone for text recognition, modified from `ResNet.

    <https://arxiv.org/pdf/1512.03385.pdf>`_ and
    `<https://github.com/FangShancheng/ABINet>`_

    Args:
        base_channels (int): Number of channels of input image tensor.
        stem_channels (int): Number of stem channels.
        arch_settings  (list[int]): List of BasicBlock number for each stage.
        strides (Sequence[int]): Strides of the first block of each stage.
        out_indices (None | Sequence[int]): Indices of output stages.
        last_stage_pool (bool): If True, add `MaxPool2d` layer to last stage.
    """

    def __init__(self,
                 base_channels=3,
                 stem_channels=32,
                 arch_settings=[3, 4, 6, 6, 3],
                 strides=[2, 1, 2, 1, 1],
                 out_indices=None,
                 last_stage_pool=False,
                 init_cfg=[
                     dict(type='Xavier', layer='Conv2d'),
                     dict(type='Constant', val=1, layer='BatchNorm2d')
                 ]):
        super().__init__(init_cfg=init_cfg)
        assert isinstance(base_channels, int)
        assert isinstance(stem_channels, int)
        assert utils.is_type_list(arch_settings, int)
        assert utils.is_type_list(strides, int)
        assert len(arch_settings) == len(strides)
        assert out_indices is None or isinstance(out_indices, (list, tuple))
        assert isinstance(last_stage_pool, bool)

        self.out_indices = out_indices
        self.last_stage_pool = last_stage_pool

        self.conv1 = nn.Conv2d(
            base_channels, stem_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(stem_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.layers = [
            self._make_layer(stem_channels, stem_channels, arch_settings[0],
                             strides[0])
        ]
        for i in range(1, len(arch_settings)):
            self.layers.append(
                self._make_layer(stem_channels * 2**(i - 1),
                                 stem_channels * 2**i, arch_settings[i],
                                 strides[i]))
        self.layers = Sequential(*self.layers)

    def _make_layer(self, input_channels, output_channels, blocks, stride=1):
        layers = []
        downsample = None
        if stride != 1 or input_channels != output_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    input_channels, output_channels, 1, stride, bias=False),
                nn.BatchNorm2d(output_channels),
            )
        layers.append(
            BasicBlock(
                input_channels,
                output_channels,
                use_conv1x1=True,
                stride=stride,
                downsample=downsample))
        input_channels = output_channels
        for _ in range(1, blocks):
            layers.append(
                BasicBlock(input_channels, output_channels, use_conv1x1=True))

        return Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        outs = []
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            outs.append(x)

        if self.out_indices is not None:
            return tuple([outs[i] for i in self.out_indices])

        return x

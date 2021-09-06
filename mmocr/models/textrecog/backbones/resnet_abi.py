# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner import BaseModule, Sequential

import mmocr.utils as utils
from mmocr.models.builder import BACKBONES
from mmocr.models.textrecog.layers import BasicBlock


@BACKBONES.register_module()
class ResNetABI(BaseModule):
    """Implement ResNet backbone for text recognition, modified from
      `ResNet <https://arxiv.org/pdf/1512.03385.pdf>`_ and
      `<https://github.com/FangShancheng/ABINet>`_
    Args:
        base_channels (int): Number of channels of input image tensor.
        layers (list[int]): List of BasicBlock number for each stage.
        channels (list[int]): List of out_channels of Conv2d layer.
        out_indices (None | Sequence[int]): Indices of output stages.
        last_stage_pool (bool): If True, add `MaxPool2d` layer to last stage.
    """

    def __init__(self,
                 base_channels=3,
                 arch_settings=[3, 4, 6, 6, 3],
                 channels=[32, 32, 64, 128, 256, 512],
                 strides=[1, 2, 1, 2, 1, 1],
                 out_indices=None,
                 last_stage_pool=False,
                 init_cfg=[
                     dict(type='Xavier', layer='Conv2d'),
                     dict(type='Constant', val=1, layer='BatchNorm2d')
                 ]):
        super().__init__(init_cfg=init_cfg)
        assert isinstance(base_channels, int)
        assert utils.is_type_list(arch_settings, int)
        assert utils.is_type_list(channels, int)
        assert utils.is_type_list(strides, int)
        assert len(arch_settings) == len(channels) - 1
        assert len(arch_settings) == len(strides) - 1
        assert out_indices is None or isinstance(out_indices, (list, tuple))
        assert isinstance(last_stage_pool, bool)

        self.out_indices = out_indices
        self.last_stage_pool = last_stage_pool

        self.conv1 = nn.Conv2d(
            base_channels,
            channels[0],
            kernel_size=3,
            stride=strides[0],
            padding=1)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu1 = nn.ReLU(inplace=True)

        self.layers = [
            self._make_layer(channels[i], channels[i + 1], arch_settings[i],
                             strides[i]) for i in range(1, len(arch_settings))
        ]

    def _make_layer(self, input_channels, output_channels, blocks, stride=1):
        layers = []
        downsample = stride != 1 or input_channels != output_channels
        layers.append(
            BasicBlock(
                input_channels,
                output_channels,
                stride=stride,
                downsample=downsample))
        input_channels = output_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(input_channels, output_channels))

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

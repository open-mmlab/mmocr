# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmengine.model import BaseModule, Sequential

import mmocr.utils as utils
from mmocr.models.textrecog.layers import BasicBlock
from mmocr.registry import MODELS


@MODELS.register_module()
class ResNetABI(BaseModule):
    """Implement ResNet backbone for text recognition, modified from `ResNet.

    <https://arxiv.org/pdf/1512.03385.pdf>`_ and
    `<https://github.com/FangShancheng/ABINet>`_

    Args:
        in_channels (int): Number of channels of input image tensor.
        stem_channels (int): Number of stem channels.
        base_channels (int): Number of base channels.
        arch_settings  (list[int]): List of BasicBlock number for each stage.
        strides (Sequence[int]): Strides of the first block of each stage.
        out_indices (None | Sequence[int]): Indices of output stages. If not
            specified, only the last stage will be returned.
        last_stage_pool (bool): If True, add `MaxPool2d` layer to last stage.
    """

    def __init__(self,
                 in_channels=3,
                 stem_channels=32,
                 base_channels=32,
                 arch_settings=[3, 4, 6, 6, 3],
                 strides=[2, 1, 2, 1, 1],
                 out_indices=None,
                 last_stage_pool=False,
                 init_cfg=[
                     dict(type='Xavier', layer='Conv2d'),
                     dict(type='Constant', val=1, layer='BatchNorm2d')
                 ]):
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, int)
        assert isinstance(stem_channels, int)
        assert utils.is_type_list(arch_settings, int)
        assert utils.is_type_list(strides, int)
        assert len(arch_settings) == len(strides)
        assert out_indices is None or isinstance(out_indices, (list, tuple))
        assert isinstance(last_stage_pool, bool)

        self.out_indices = out_indices
        self.last_stage_pool = last_stage_pool
        self.block = BasicBlock
        self.inplanes = stem_channels

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        planes = base_channels
        for i, num_blocks in enumerate(arch_settings):
            stride = strides[i]
            res_layer = self._make_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                blocks=num_blocks,
                stride=stride)
            self.inplanes = planes * self.block.expansion
            planes *= 2
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        layers = []
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers.append(
            block(
                inplanes,
                planes,
                use_conv1x1=True,
                stride=stride,
                downsample=downsample))
        inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, use_conv1x1=True))

        return Sequential(*layers)

    def _make_stem_layer(self, in_channels, stem_channels):
        self.conv1 = nn.Conv2d(
            in_channels, stem_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(stem_channels)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x (Tensor): Image tensor of shape :math:`(N, 3, H, W)`.

        Returns:
            Tensor or list[Tensor]: Feature tensor. Its shape depends on
            ResNetABI's config. It can be a list of feature outputs at specific
            layers if ``out_indices`` is specified.
        """

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if self.out_indices and i in self.out_indices:
                outs.append(x)

        return tuple(outs) if self.out_indices else x

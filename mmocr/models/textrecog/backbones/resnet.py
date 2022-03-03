# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner import BaseModule, Sequential

import mmocr.utils as utils
from mmocr.models.builder import BACKBONES
from mmocr.models.textrecog.layers import BasicBlock


class ResNetOCR(BaseModule):
    """Implement general ResNet backbone for text recognition
       Supporting: ResNet31, ResNet45, ResNet31_Master

    Args:
        in_channels (int): Number of channels of input image tensor.

        stem_channels (list[int]): List of channels in each layer of stem.e.g.
        [64, 128] stands for 64 channels in the first layer and 128 channels
        in the second layer of stem
        arch_layers (list[int]): List of Block number for each stage.
        arch_channels (list[int]): List of channels for each stage.
        use_conv1x1 (bool): Using 1x1 convolution in each stage
        strides (Sequence[int]): Strides of the first block of each stage.
        out_indices (None | Sequence[int]): Indices of output stages. If not
            specified, only the last stage will be returned.
    """

    def __init__(self,
                 in_channels,
                 stem_channels,
                 arch_layers,
                 arch_channels,
                 use_conv1x1,
                 strides,
                 out_indices=None,
                 init_cfg=[
                     dict(type='Xavier', layer='Conv2d'),
                     dict(type='Constant', val=1, layer='BatchNorm2d'),
                 ]):
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, int)
        assert isinstance(stem_channels, int) or utils.is_type_list(
            stem_channels, int)
        assert utils.is_type_list(arch_layers, int)
        assert utils.is_type_list(arch_channels, int)
        assert utils.is_type_list(strides, tuple)
        assert len(arch_layers) == len(arch_channels) == len(strides)
        assert out_indices is None or isinstance(out_indices, (list, tuple))

        self.out_indices = out_indices
        if isinstance(stem_channels, int):
            self.inplanes = stem_channels
        else:
            self.inplanes = stem_channels[-1]
        self.block = BasicBlock
        self.use_conv1x1 = use_conv1x1
        self._make_stem_layer(in_channels, stem_channels)

        self.res_stages = []
        for i, num_blocks in enumerate(arch_layers):
            stride = strides[i]
            channel = arch_channels[i]
            res_stage = self._make_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=channel,
                blocks=num_blocks,
                stride=stride)
            self.inplanes = channel
            stage_name = f'stage{i + 1}'
            self.add_module(stage_name, res_stage)
            self.res_stages.append(stage_name)

    def _make_layer(self, block, inplanes, planes, blocks, stride):
        layers = []
        downsample = None
        if stride[0] != 1 or stride[1] != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers.append(
            block(
                inplanes,
                planes,
                stride=stride,
                use_conv1x1=self.use_conv1x1,
                downsample=downsample))
        inplanes = planes
        for _ in range(1, blocks):
            layers.append(
                block(inplanes, planes, use_conv1x1=self.use_conv1x1))

        return Sequential(*layers)

    def _make_stem_layer(self, in_channels, stem_channels):
        if isinstance(stem_channels, int):
            self.stem_layers = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False), nn.BatchNorm2d(stem_channels),
                nn.ReLU(inplace=True))
        else:
            stem_layers = []
            for i, channels in enumerate(stem_channels):
                stem_layer = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False), nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True))
                in_channels = channels
                stem_layers.append(stem_layer)
            self.stem_layers = Sequential(*stem_layers)


@BACKBONES.register_module()
class ResNet45_abi(ResNetOCR):
    """Implement ResNet45 for ABINet
    Args:
        in_channels (int): Number of channels of input image tensor.

        stem_channels (list[int]): List of channels in each layer of stem.e.g.
        [64, 128] stands for 64 channels in the first layer and 128 channels
        in the second layer of stem
        arch_layers (list[int]): List of Block number for each stage.
        arch_channels (list[int]): List of channels for each stage.
        use_conv1x1 (bool): Using 1x1 convolution in each stage
        strides (Sequence[int]): Strides of the first block of each stage.
        out_indices (None | Sequence[int]): Indices of output stages. If not
            specified, only the last stage will be returned.
    """

    def __init__(self,
                 in_channels=3,
                 stem_channels=32,
                 arch_layers=[3, 4, 6, 6, 3],
                 arch_channels=[32, 64, 128, 256, 512],
                 use_conv1x1=True,
                 strides=[(2, 2), (1, 1), (2, 2), (1, 1), (1, 1)],
                 out_indices=None):
        super().__init__(in_channels, stem_channels, arch_layers,
                         arch_channels, use_conv1x1, strides, out_indices)

    def forward(self, x):
        """
        Args:
            x (Tensor): Image tensor of shape :math:`(N, 3, H, W)`.

        Returns:
            Tensor or list[Tensor]: Feature tensor. Its shape depends on
            ResNetABI's config. It can be a list of feature outputs at specific
            layers if ``out_indices`` is specified.

            output (Tensor): shape :math: `(N, 512, H/4, W/4)`
        """
        x = self.stem_layers(x)

        outs = []
        for i, layer_name in enumerate(self.res_stages):
            res_stage = getattr(self, layer_name)
            x = res_stage(x)
            if self.out_indices and i in self.out_indices:
                outs.append(x)

        return tuple(outs) if self.out_indices else x


@BACKBONES.register_module()
class ResNet45_aster(ResNetOCR):
    """Implement ResNet45 for ABINet
    Args:
        in_channels (int): Number of channels of input image tensor.

        stem_channels (list[int]): List of channels in each layer of stem.e.g.
        [64, 128] stands for 64 channels in the first layer and 128 channels
        in the second layer of stem
        arch_layers (list[int]): List of Block number for each stage.
        arch_channels (list[int]): List of channels for each stage.
        use_conv1x1 (bool): Using 1x1 convolution in each stage
        strides (Sequence[int]): Strides of the first block of each stage.
        out_indices (None | Sequence[int]): Indices of output stages. If not
            specified, only the last stage will be returned.
    """

    def __init__(self,
                 in_channels=3,
                 stem_channels=32,
                 arch_layers=[3, 4, 6, 6, 3],
                 arch_channels=[32, 64, 128, 256, 512],
                 use_conv1x1=True,
                 strides=[(2, 2), (2, 2), (2, 1), (2, 1), (2, 1)],
                 out_indices=None):
        super().__init__(in_channels, stem_channels, arch_layers,
                         arch_channels, use_conv1x1, strides, out_indices)

    def forward(self, x):
        """
        Args:
            x (Tensor): Image tensor of shape :math:`(N, 3, H, W)`.

        Returns:
            Tensor or list[Tensor]: Feature tensor. Its shape depends on
            ResNetABI's config. It can be a list of feature outputs at specific
            layers if ``out_indices`` is specified.
            output (Tensor): shape :math: `(N, 512, H/32, W/4)`
        """
        x = self.stem_layers(x)

        outs = []
        for i, layer_name in enumerate(self.res_stages):
            res_stage = getattr(self, layer_name)
            x = res_stage(x)
            if self.out_indices and i in self.out_indices:
                outs.append(x)

        return tuple(outs) if self.out_indices else x


@BACKBONES.register_module()
class ResNet31(ResNetOCR):
    """Implement ResNet45 for ABINet
    Args:
        in_channels (int): Number of channels of input image tensor.

        stem_channels (list[int]): List of channels in each layer of stem.e.g.
        [64, 128] stands for 64 channels in the first layer and 128 channels
        in the second layer of stem
        arch_layers (list[int]): List of Block number for each stage.
        arch_channels (list[int]): List of channels for each stage.
        use_conv1x1 (bool): Using 1x1 convolution in each stage
        strides (Sequence[int]): Strides of the first block of each stage.
        out_indices (None | Sequence[int]): Indices of output stages. If not
            specified, only the last stage will be returned.
        stage4_pool_cfg (dict): Dictionary to construct and configure
            pooling layer in stage 4.
        last_stage_pool (bool): If True, add `MaxPool2d` layer to last stage.
    """

    def __init__(self,
                 in_channels=3,
                 stem_channels=[64, 128],
                 arch_layers=[1, 2, 5, 3],
                 arch_channels=[256, 256, 512, 512],
                 use_conv1x1=False,
                 strides=[(1, 1), (1, 1), (1, 1), (1, 1)],
                 out_indices=None,
                 stage4_pool_cfg=dict(kernel_size=(2, 1), stride=(2, 1)),
                 last_stage_pool=False):
        super().__init__(in_channels, stem_channels, arch_layers,
                         arch_channels, use_conv1x1, strides, out_indices)

        self.pooling_layers = []
        self.add_conv_layers = []
        self.last_stage_pool = last_stage_pool
        # Additional layers
        for i in range(len(arch_layers)):
            if i in [1, 2]:
                self.add_module(f'pooling{i + 1}', nn.MaxPool2d(2, 2))
                self.pooling_layers.append(f'pooling{i + 1}')
            elif i == 3:
                self.add_module(f'pooling{i + 1}',
                                nn.MaxPool2d(**stage4_pool_cfg))
                self.pooling_layers.append(f'pooling{i + 1}')
            else:
                if last_stage_pool:
                    self.add_module(f'pooling{i + 1}', nn.MaxPool2d(2, 2))
                    self.pooling_layers.append(f'pooling{i + 1}')

            self.add_module(
                f'add_conv{i + 1}',
                nn.Sequential(
                    nn.Conv2d(
                        arch_channels[i],
                        arch_channels[i],
                        kernel_size=3,
                        stride=1,
                        padding=1), nn.BatchNorm2d(arch_channels[i]),
                    nn.ReLU(inplace=True)))
            self.add_conv_layers.append(f'add_conv{i + 1}')

    def forward(self, x):
        """
        Args:
            x (Tensor): Image tensor of shape :math:`(N, 3, H, W)`.

        Returns:
            Tensor or list[Tensor]: Feature tensor. Its shape depends on
            ResNetABI's config. It can be a list of feature outputs at specific
            layers if ``out_indices`` is specified.

            output (Tensor): shape :math: `(N, 512, H/32, W/4)`
        """
        x = self.stem_layers(x)

        outs = []
        for i, layer_name in enumerate(self.res_stages):
            if i != 3 or self.last_stage_pool:
                pooling_layer = getattr(self, self.pooling_layers[i])
            res_stage = getattr(self, layer_name)
            add_conv_layer = getattr(self, self.add_conv_layers[i])

            if i != 3 or self.last_stage_pool:
                x = pooling_layer(x)
            x = res_stage(x)
            x = add_conv_layer(x)
            if self.out_indices and i in self.out_indices:
                outs.append(x)

        return tuple(outs) if self.out_indices else x

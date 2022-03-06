# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner import BaseModule, Sequential

import mmocr.utils as utils
from mmocr.models.builder import BACKBONES


@BACKBONES.register_module()
class ResNetOCR(BaseModule):
    """Implement general ResNet backbone for text recognition
       Supporting: ResNet31, ResNet45, ResNet31_Master

    Args:
        in_channels (int): Number of channels of input image tensor.

        stem_channels (list[int]): List of channels in each layer of stem.e.g.
        [64, 128] stands for 64 channels in the first layer and 128 channels
        in the second layer of stem
        stage (class): BasicStage, Stage_31, Stage_Master
        arch_layers (list[int]): List of Block number for each stage.
        arch_channels (list[int]): List of channels for each stage.
        use_conv1x1 (bool): Using 1x1 convolution in each stage
        strides (Sequence[int]): Strides of the first block of each stage.
        out_indices (None | Sequence[int]): Indices of output stages. If not
            specified, only the last stage will be returned.
        pool_cfg (list[dict])
    """

    def __init__(self,
                 in_channels,
                 stem_channels,
                 stage,
                 arch_layers,
                 arch_channels,
                 use_conv1x1,
                 strides,
                 out_indices=None,
                 pool_cfg=None,
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
        self.stage = stage
        self._make_stem_layer(in_channels, stem_channels)

        self.res_stages = []
        for i, num_blocks in enumerate(arch_layers):
            stride = strides[i]
            channel = arch_channels[i]
            res_stage = self._make_layer(
                stage=self.stage,
                inplanes=self.inplanes,
                planes=channel,
                use_conv1x1=use_conv1x1,
                blocks=num_blocks,
                stride=stride,
                pool_cfg=pool_cfg[i] if pool_cfg else None)
            self.inplanes = channel
            stage_name = f'stage{i + 1}'
            self.add_module(stage_name, res_stage)
            self.res_stages.append(stage_name)

    def _make_layer(self, stage, inplanes, planes, use_conv1x1, blocks, stride,
                    pool_cfg):
        layers = []
        downsample = None
        if stride[0] != 1 or stride[1] != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        if pool_cfg:
            layers.append(
                stage(
                    inplanes,
                    planes,
                    use_conv1x1,
                    stride=stride,
                    downsample=downsample,
                    pool_cfg=pool_cfg))
        else:
            layers.append(
                stage(
                    inplanes,
                    planes,
                    use_conv1x1,
                    stride=stride,
                    downsample=downsample))

        inplanes = planes
        for _ in range(1, blocks):
            layers.append(stage(inplanes, planes, use_conv1x1))

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
            self.inplanes = stem_channels
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
            self.inplanes = stem_channels[-1]

    def forward(self, x):
        """
        Args:p
            x (Tensor): Image tensor of shae :math:`(N, 3, H, W)`.

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

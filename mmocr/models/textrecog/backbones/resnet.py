# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
from mmcv.cnn import ConvModule, build_plugin_layer
from mmcv.runner import BaseModule, Sequential

import mmocr.utils as utils
from mmocr.models.textrecog.layers import BasicBlock
from mmocr.registry import MODELS


@MODELS.register_module()
class ResNet(BaseModule):
    """
    Args:
        in_channels (int): Number of channels of input image tensor.
        stem_channels (list[int]): List of channels in each stem layer. E.g.,
            [64, 128] stands for 64 and 128 channels in the first and second
            stem layers.
        block_cfgs (dict): Configs of block
        arch_layers (list[int]): List of Block number for each stage.
        arch_channels (list[int]): List of channels for each stage.
        strides (Sequence[int] or Sequence[tuple]): Strides of the first block
            of each stage.
        out_indices (Sequence[int], optional): Indices of output stages. If not
            specified, only the last stage will be returned.
        plugins (dict, optional): Configs of stage plugins
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels: int,
                 stem_channels: List[int],
                 block_cfgs: dict,
                 arch_layers: List[int],
                 arch_channels: List[int],
                 strides: Union[List[int], List[Tuple]],
                 out_indices: Optional[List[int]] = None,
                 plugins: Optional[Dict] = None,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = [
                     dict(type='Xavier', layer='Conv2d'),
                     dict(type='Constant', val=1, layer='BatchNorm2d'),
                 ]):
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, int)
        assert isinstance(stem_channels, int) or utils.is_type_list(
            stem_channels, int)
        assert utils.is_type_list(arch_layers, int)
        assert utils.is_type_list(arch_channels, int)
        assert utils.is_type_list(strides, tuple) or utils.is_type_list(
            strides, int)
        assert len(arch_layers) == len(arch_channels) == len(strides)
        assert out_indices is None or isinstance(out_indices, (list, tuple))

        self.out_indices = out_indices
        self._make_stem_layer(in_channels, stem_channels)
        self.num_stages = len(arch_layers)
        self.use_plugins = False
        self.arch_channels = arch_channels
        self.res_layers = []
        if plugins is not None:
            self.plugin_ahead_names = []
            self.plugin_after_names = []
            self.use_plugins = True
        for i, num_blocks in enumerate(arch_layers):
            stride = strides[i]
            channel = arch_channels[i]

            if self.use_plugins:
                self._make_stage_plugins(plugins, stage_idx=i)

            res_layer = self._make_layer(
                block_cfgs=block_cfgs,
                inplanes=self.inplanes,
                planes=channel,
                blocks=num_blocks,
                stride=stride,
            )
            self.inplanes = channel
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

    def _make_layer(self, block_cfgs: Dict, inplanes: int, planes: int,
                    blocks: int, stride: int) -> Sequential:
        """Build resnet layer.

        Args:
            block_cfgs (dict): Configs of blocks.
            inplanes (int): Number of input channels.
            planes (int): Number of output channels.
            blocks (int): Number of blocks.
            stride (int): Stride of the first block.

        Returns:
            Sequential: A sequence of blocks.
        """
        layers = []
        downsample = None
        block_cfgs_ = block_cfgs.copy()
        if isinstance(stride, int):
            stride = (stride, stride)

        if stride[0] != 1 or stride[1] != 1 or inplanes != planes:
            downsample = ConvModule(
                inplanes,
                planes,
                1,
                stride,
                norm_cfg=dict(type='BN'),
                act_cfg=None)

        if block_cfgs_['type'] == 'BasicBlock':
            block = BasicBlock
            block_cfgs_.pop('type')
        else:
            raise ValueError('{} not implement yet'.format(block['type']))

        layers.append(
            block(
                inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                **block_cfgs_))
        inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, **block_cfgs_))

        return Sequential(*layers)

    def _make_stem_layer(self, in_channels: int,
                         stem_channels: Union[int, List[int]]) -> None:
        """Make stem layers.

        Args:
            in_channels (int): Number of input channels.
            stem_channels (list[int] or int): List of channels in each stem
                layer. If int, only one stem layer will be created.
        """
        if isinstance(stem_channels, int):
            stem_channels = [stem_channels]
        stem_layers = []
        for _, channels in enumerate(stem_channels):
            stem_layer = ConvModule(
                in_channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU'))
            in_channels = channels
            stem_layers.append(stem_layer)
        self.stem_layers = Sequential(*stem_layers)
        self.inplanes = stem_channels[-1]

    def _make_stage_plugins(self, plugins: List[Dict], stage_idx: int) -> None:
        """Make plugins for ResNet ``stage_idx``th stage.

        Currently we support inserting ``nn.Maxpooling``,
        ``mmcv.cnn.Convmodule``into the backbone. Originally designed
        for ResNet31-like architectures.

        Examples:
            >>> plugins=[
            ...     dict(cfg=dict(type="Maxpooling", arg=(2,2)),
            ...          stages=(True, True, False, False),
            ...          position='before_stage'),
            ...     dict(cfg=dict(type="Maxpooling", arg=(2,1)),
            ...          stages=(False, False, True, Flase),
            ...          position='before_stage'),
            ...     dict(cfg=dict(
            ...              type='ConvModule',
            ...              kernel_size=3,
            ...              stride=1,
            ...              padding=1,
            ...              norm_cfg=dict(type='BN'),
            ...              act_cfg=dict(type='ReLU')),
            ...          stages=(True, True, True, True),
            ...          position='after_stage')]

        Suppose ``stage_idx=1``, the structure of stage would be:

        .. code-block:: none

            Maxpooling -> A set of Basicblocks -> ConvModule

        Args:
            plugins (list[dict]): List of plugin configs to build.
            stage_idx (int): Index of stage to build
        """
        in_channels = self.arch_channels[stage_idx]
        self.plugin_ahead_names.append([])
        self.plugin_after_names.append([])
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            position = plugin.pop('position', None)
            assert stages is None or len(stages) == self.num_stages
            if stages[stage_idx]:
                if position == 'before_stage':
                    name, layer = build_plugin_layer(
                        plugin['cfg'],
                        f'_before_stage_{stage_idx+1}',
                        in_channels=in_channels,
                        out_channels=in_channels)
                    self.plugin_ahead_names[stage_idx].append(name)
                    self.add_module(name, layer)
                elif position == 'after_stage':
                    name, layer = build_plugin_layer(
                        plugin['cfg'],
                        f'_after_stage_{stage_idx+1}',
                        in_channels=in_channels,
                        out_channels=in_channels)
                    self.plugin_after_names[stage_idx].append(name)
                    self.add_module(name, layer)
                else:
                    raise ValueError('uncorrect plugin position')

    def forward_plugin(self, x: torch.Tensor,
                       plugin_name: List[str]) -> torch.Tensor:
        """Forward tensor through plugin.

        Args:
            x (torch.Tensor): Input tensor.
            plugin_name (list[str]): Name of plugins.

        Returns:
            torch.Tensor: Output tensor.
        """
        out = x
        for name in plugin_name:
            out = getattr(self, name)(out)
        return out

    def forward(self,
                x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Args: x (Tensor): Image tensor of shape :math:`(N, 3, H, W)`.

        Returns:
            Tensor or list[Tensor]: Feature tensor. It can be a list of
            feature outputs at specific layers if ``out_indices`` is specified.
        """
        x = self.stem_layers(x)

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            if not self.use_plugins:
                x = res_layer(x)
                if self.out_indices and i in self.out_indices:
                    outs.append(x)
            else:
                x = self.forward_plugin(x, self.plugin_ahead_names[i])
                x = res_layer(x)
                x = self.forward_plugin(x, self.plugin_after_names[i])
                if self.out_indices and i in self.out_indices:
                    outs.append(x)

        return tuple(outs) if self.out_indices else x

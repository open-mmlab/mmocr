# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def conv1x1(in_planes, out_planes):
    """1x1 convolution with padding."""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)


class BasicBlock_New(nn.Module):

    def __init__(
        self,
        cfg,
        inplanes,
        planes,
        stride=1,
        downsample=None,
    ):
        super(BasicBlock_New, self).__init__()

        if 'use_conv1x1' in cfg and cfg['use_conv1x1'] is True:
            self.conv1 = conv1x1(inplanes, planes)
            self.conv2 = conv3x3(planes, planes, stride)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.conv2 = conv3x3(planes, planes)

        self.plugin = None
        if 'plugins' in cfg and cfg['plugins'] is not None:
            self.plugin = self._make_block_plugins(cfg['plugins'])
        self.planes = planes
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def _creat_plugin(self, plugin_cfg):
        if plugin_cfg['type'] == 'Maxpooling':
            return nn.MaxPool2d(plugin_cfg['arg'])
        elif plugin_cfg['type'] == 'Conv':
            return nn.Sequential(
                nn.Conv2d(
                    self.planes, self.planes, **plugin_cfg['arg'], bias=False),
                nn.BatchNorm2d(self.planes), nn.ReLU(inplace=True))
        elif plugin_cfg['type'] == 'GCA':
            raise Exception('{} not implemented yet'.format(plugin_cfg['arg']))

    def _make_block_plugins(self, plugins):
        """Make plugins for ResNet ``stage_idx`` th stage.
        Currently we support to insert ``nn.Maxpooling``,
        ``nn.conv2d``into the backbone. This is designed for ResNet31
        architecture likes
        An example of plugins format could be:
        Examples:
            >>> plugins=[
            ...     dict(cfg=dict(type="Maxpooling", stride=(2,2)),
            ...          position='before_conv1'),
            ...     dict(cfg=dict(type="Maxpooling", stride=(2,1)),
            ...          position='before_conv2'),
            ...     dict(cfg=dict(type="Conv",
            ...          args=(dict(kernel_size=3, stride=1, padding=1))),
            ...          position=('after_conv2')
            ... ]
        Suppose ``stage_idx=1``, the structure of stage would be:
        .. code-block:: none
            Maxpooling-> Basicbloks * blocks[i] -> Conv
        Suppose 'stage_idx=1', the structure of blocks in the stage would be:
        If stages is missing, the plugin would be applied to all stages.
        Args:
            plugins (list[dict]): List of plugins cfg to build. The postfix is
                required if multiple same type plugins are inserted.
            stage_idx (int): Index of stage to build
        Returns:
            list[dict]: Plugins for current stage
        """

        exist_position = []
        for plugin in plugins:
            plugin = plugin.copy()
            position = plugin.pop('position', None)
            assert position in exist_position
            self.plugin_before_conv1 = None
            self.plugin_before_conv2 = None
            self.plugin_after_conv2 = None
            if position == 'before_conv1':
                self.plugin_before_conv1 = self._creat_plugin(plugin['cfg'])
            elif position == 'before_conv2':
                self.plugin_before_conv2 = self._creat_plugin(plugin['cfg'])
            elif position == 'after_conv2':
                self.plugin_after_conv2 = self._creat_plugin(plugin['cfg'])
            else:
                raise Exception('uncorrect plugin position')

    def forward(self, x):
        residual = x

        if self.plugin:
            if self.plugin_before_conv1:
                out = self.plugin_before_conv1(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.plugin:
            if self.plugin_before_conv2:
                out = self.plugin_before_conv2(x)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        if self.plugin:
            if self.plugin_after_conv2:
                out = self.plugin_after_conv2(x)

        return out

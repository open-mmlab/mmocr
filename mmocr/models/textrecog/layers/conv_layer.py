# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import build_plugin_layer


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


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 use_conv1x1=False,
                 plugins=None):
        super(BasicBlock, self).__init__()

        if use_conv1x1:
            self.conv1 = conv1x1(inplanes, planes)
            self.conv2 = conv3x3(planes, planes * self.expansion, stride)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.conv2 = conv3x3(planes, planes * self.expansion)

        self.with_plugins = False
        if plugins:
            if isinstance(plugins, dict):
                plugins = [plugins]
            self.with_plugins = True
            # collect plugins for conv1/conv2/
            self.before_conv1_plugin = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'before_conv1'
            ]
            self.after_conv1_plugin = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv1'
            ]
            self.after_conv2_plugin = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv2'
            ]
            self.after_shortcut_plugin = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_shortcut'
            ]

        self.planes = planes
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

        if self.with_plugins:
            self.before_conv1_plugin_names = self.make_block_plugins(
                inplanes, self.before_conv1_plugin)
            self.after_conv1_plugin_names = self.make_block_plugins(
                planes, self.after_conv1_plugin)
            self.after_conv2_plugin_names = self.make_block_plugins(
                planes, self.after_conv2_plugin)
            self.after_shortcut_plugin_names = self.make_block_plugins(
                planes, self.after_shortcut_plugin)

    def make_block_plugins(self, in_channels, plugins):
        """make plugins for block.

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.
        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin,
                in_channels=in_channels,
                out_channels=in_channels,
                postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(x)
        return out

    def forward(self, x):
        if self.with_plugins:
            x = self.forward_plugin(x, self.before_conv1_plugin_names)
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.with_plugins:
            out = self.forward_plugin(out, self.after_conv1_plugin_names)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.with_plugins:
            out = self.forward_plugin(out, self.after_conv2_plugin_names)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        if self.with_plugins:
            out = self.forward_plugin(out, self.after_shortcut_plugin_names)

        return out


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

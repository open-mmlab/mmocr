# Copyright (c) OpenMMLab. All rights reserved.

import torch.nn as nn
from mmdet.models.backbones import ResNet
from mmdet.models.backbones.resnet import Bottleneck

from mmocr.registry import MODELS


class CLIPBottleneck(Bottleneck):
    """Bottleneck for CLIPResNet.

    It is a variant Bottleneck used in the variant ResNet of CLIP. After the
    second convolution layer, there is an additional average pooling layer with
    kernel_size 2 and stride 2, which is added in the plug-in manner when the
    input stride > 1. The stride of each convolution layer is always set to 1.
    """

    def __init__(self, **kwargs):
        stride = kwargs.get('stride', 1)
        kwargs['stride'] = 1
        plugins = kwargs.get('plugins', None)
        if stride > 1:
            if plugins is None:
                plugins = []

            plugins.insert(
                0,
                dict(
                    cfg=dict(type='mmocr.AvgPool2d', kernel_size=2),
                    position='after_conv2'))
            kwargs['plugins'] = plugins
        super().__init__(**kwargs)


@MODELS.register_module()
class CLIPResNet(ResNet):
    """Implement the variant ResNet used in `oCLIP.

    <https://github.com/bytedance/oclip>`_

    It is also the official structure in `CLIP
    <https://github.com/openai/CLIP>`_.

    Compared with ResNetV1d structure, CLIPResNet replaces the
    max pooling layer with an average pooling layer at the end
    of the input stem.

    In the Bottleneck of CLIPResNet, after the second convolution
    layer, there is an additional average pooling layer with
    kernel_size 2 and stride 2, which is added in the plug-in
    manner when the input stride > 1.
    The stride of each convolution layer is always set to 1.
    """
    arch_settings = {
        50: (CLIPBottleneck, (3, 4, 6, 3)),
    }

    def __init__(self,
                 depth=50,
                 strides=(1, 2, 2, 2),
                 deep_stem=True,
                 avg_down=True,
                 **kwargs):
        super().__init__(
            depth=depth,
            strides=strides,
            deep_stem=deep_stem,
            avg_down=avg_down,
            **kwargs)

    def _make_stem_layer(self, in_channels, stem_channels):
        """Build stem layer for CLIPResNet used in `CLIP.

        <https://github.com/openai/CLIP>`_.
        It uses an average pooling layer rather than a max pooling
        layer at the end of the input stem.

        Args:
            in_channels (int): Number of input channels.
            stem_channels (int): Number of output channels.
        """
        super()._make_stem_layer(in_channels, stem_channels)
        if self.deep_stem:
            self.maxpool = nn.AvgPool2d(kernel_size=2)

# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import PLUGIN_LAYERS


@PLUGIN_LAYERS.register_module()
class Maxpool2d(nn.Module):
    """A wrapper around nn.Maxpool2d().

    Args:
        kernel_size (int or tuple(int)): Kernel size for max pooling layer
        stride (int or tuple(int)): Stride for max pooling layer
        padding (int or tuple(int)): Padding for pooling layer
    """

    def __init__(self, kernel_size, stride, padding=0, **kwargs):
        super(Maxpool2d, self).__init__()
        self.model = nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input feature map

        Returns:
            Tensor: The tensor after Maxpooling layer.
        """
        return self.model(x)


@PLUGIN_LAYERS.register_module()
class GCAModule(nn.Module):
    """GCAModule in MASTER
    Args:
        in_channels: Channels of input tensor
        ratio: Scale ratio of in_channels
        headers: Nums of attention head
        pooling_type: Spatial pooling type
        is_att_scale: Scale attention map or not
        fusion_type: Fusion type of input and context
    """

    def __init__(self,
                 in_channels,
                 ratio,
                 headers,
                 pooling_type='att',
                 is_att_scale=False,
                 fusion_type='channel_add',
                 **kwargs):
        super(GCAModule, self).__init__()

        assert pooling_type in ['avg', 'att']
        assert fusion_type in ['channel_add', 'channel_mul', 'channel_concat']

        # in_channels must be divided by headers evenly
        assert in_channels % headers == 0 and in_channels >= 8

        self.headers = headers
        self.in_channels = in_channels
        self.ratio = ratio
        self.planes = int(in_channels * ratio)
        self.pooling_type = pooling_type
        self.fusion_type = fusion_type
        self.is_att_scale = is_att_scale
        self.single_header_inplanes = int(in_channels / headers)

        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(
                self.single_header_inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if fusion_type == 'channel_add':
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.in_channels, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]), nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.in_channels, kernel_size=1))
        elif fusion_type == 'channel_concat':
            self.channel_concat_conv = nn.Sequential(
                nn.Conv2d(self.in_channels, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]), nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.in_channels, kernel_size=1))
            # for concat
            self.cat_conv = nn.Conv2d(
                2 * self.in_channels, self.in_channels, kernel_size=1)
        elif fusion_type == 'channel_mul':
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.in_channels, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]), nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.in_channels, kernel_size=1))

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            # [N*headers, C', H , W] C = headers * C'
            x = x.view(batch * self.headers, self.single_header_inplanes,
                       height, width)
            input_x = x

            # [N*headers, C', H * W] C = headers * C'
            input_x = input_x.view(batch * self.headers,
                                   self.single_header_inplanes, height * width)

            # [N*headers, 1, C', H * W]
            input_x = input_x.unsqueeze(1)
            # [N*headers, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N*headers, 1, H * W]
            context_mask = context_mask.view(batch * self.headers, 1,
                                             height * width)

            # scale variance
            if self.is_att_scale and self.headers > 1:
                context_mask = context_mask / \
                               torch.sqrt(self.single_header_inplanes)

            # [N*headers, 1, H * W]
            context_mask = self.softmax(context_mask)

            # [N*headers, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N*headers, 1, C', 1] =
            # [N*headers, 1, C', H * W] * [N*headers, 1, H * W, 1]
            context = torch.matmul(input_x, context_mask)

            # [N, headers * C', 1, 1]
            context = context.view(batch,
                                   self.headers * self.single_header_inplanes,
                                   1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)
        out = x

        if self.fusion_type == 'channel_mul':
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        elif self.fusion_type == 'channel_add':
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        else:
            # [N, C, 1, 1]
            channel_concat_term = self.channel_concat_conv(context)

            # use concat
            _, C1, _, _ = channel_concat_term.shape
            N, C2, H, W = out.shape

            out = torch.cat([out,
                             channel_concat_term.expand(-1, -1, H, W)],
                            dim=1)
            out = self.cat_conv(out)
            out = nn.functional.layer_norm(out, [self.in_channels, H, W])
            out = nn.functional.relu(out)

        return out

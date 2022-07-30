# Copyright (c) OpenMMLab. All rights reserved.
# Modified from <https://arxiv.org/abs/2205.00159>
# Adapted from <https://github.com/PaddlePaddle/PaddleOCR>
from typing import Callable

import torch.nn as nn
from mmengine.model import BaseModule


class Conv_BN_Layer(BaseModule):
    """The unit layer of conv3x3 and BatchNorm.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Number of kernel size.
        stride (int, optional): The stride for the cross-correlation.
        padding (int, optional): The amount of padding applied to the input
        bias (bool, optional): Use or not bias parameters. Defaults to True.
        groups (int, optional): The amount of group convolution.
        act (Callable, optional): The activation function.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 2,
                 padding: int = 1,
                 bias: bool = False,
                 groups: int = 1,
                 act: Callable = nn.GELU):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = act()

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): A Tensor of shape :math:`(N, C, H, W)`.

        Returns:
            Tensor: The raw logit tensor. Shape :math:`(N, C, H, W)`.
        """
        x = self.conv(x)
        x = self.norm(x)
        outputs = self.act(x)
        return outputs


class OverlapPatchEmbed(BaseModule):
    """Image to the progressive overlapping Patch Embedding.

    Args:
        input_size (int | tuple): The size of input, which will be used to
            calculate the out size.
        in_channels (int): Number of input channels.
        embed_dims (int): The dimensions of embedding.
        num_layers (int, optional): Number of Conv_BN_Layer. Defaults to 2.
    """

    def __init__(self,
                 img_size=[32, 100],
                 in_channels=3,
                 embed_dims=768,
                 num_layers=2,
                 init_cfg=None):

        super().__init__(init_cfg=init_cfg)
        """
        num_patches = (img_size[1] // (2 ** num_layers)) * \
                      (img_size[0] // (2 ** num_layers))
        """
        self.img_size = img_size
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.norm = None
        if num_layers == 2:
            self.proj = nn.Sequential(
                Conv_BN_Layer(
                    in_channels=in_channels,
                    out_channels=embed_dims // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    act=nn.GELU,
                    bias=None),
                Conv_BN_Layer(
                    in_channels=embed_dims // 2,
                    out_channels=embed_dims,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    act=nn.GELU,
                    bias=None))
        if num_layers == 3:
            self.proj = nn.Sequential(
                Conv_BN_Layer(
                    in_channels=in_channels,
                    out_channels=embed_dims // 4,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    act=nn.GELU,
                    bias=None),
                Conv_BN_Layer(
                    in_channels=embed_dims // 4,
                    out_channels=embed_dims // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    act=nn.GELU,
                    bias=None),
                Conv_BN_Layer(
                    in_channels=embed_dims // 2,
                    out_channels=embed_dims,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    act=nn.GELU,
                    bias=None))

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): A Tensor of shape :math:`(N, C, H, W)`.

        Returns:
            The feature Tensor of shape :math:`(N, HW, C)`.
        """
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model \
                ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x).flatten(2).permute(0, 2, 1)
        return x

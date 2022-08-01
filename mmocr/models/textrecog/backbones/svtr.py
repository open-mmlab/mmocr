# Copyright (c) OpenMMLab. All rights reserved.
# Modified from <https://arxiv.org/abs/2205.00159>
# Adapted from <https://github.com/PaddlePaddle/PaddleOCR>

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule


class OverlapPatchEmbed(BaseModule):
    """Image to the progressive overlapping Patch Embedding.

    Args:
        input_size (int | tuple): The size of input, which will be used to
            calculate the out size. Defaults to [32, 100].
        in_channels (int): Number of input channels. Defaults to 3.
        embed_dims (int): The dimensions of embedding. Defaults to 768.
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
                ConvModule(
                    in_channels=in_channels,
                    out_channels=embed_dims // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='GELU')),
                ConvModule(
                    in_channels=embed_dims // 2,
                    out_channels=embed_dims,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='GELU')))
        if num_layers == 3:
            self.proj = nn.Sequential(
                ConvModule(
                    in_channels=in_channels,
                    out_channels=embed_dims // 4,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='GELU')),
                ConvModule(
                    in_channels=embed_dims // 4,
                    out_channels=embed_dims // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='GELU')),
                ConvModule(
                    in_channels=embed_dims // 2,
                    out_channels=embed_dims,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='GELU')))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (Tensor): A Tensor of shape :math:`(N, C, H, W)`.

        Returns:
            Tensor: A tensor of shape math:`(N, HW_m, C)`.
        """
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model \
                ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x).flatten(2).permute(0, 2, 1)
        return x

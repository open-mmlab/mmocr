# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch import Tensor

from mmocr.models.common import MultiHeadAttention


class SATRNEncoderLayer(BaseModule):
    """Implement encoder layer for SATRN, see `SATRN.

    <https://arxiv.org/abs/1910.04396>`_.

    Args:
        d_model (int): Dimension :math:`D_m` of the input from previous model.
            Defaults to 512.
        d_inner (int): Hidden dimension of feedforward layers. Defaults to 256.
        n_head (int): Number of parallel attention heads. Defaults to 8.
        d_k (int): Dimension of the key vector. Defaults to 64.
        d_v (int): Dimension of the value vector. Defaults to 64.
        dropout (float): Dropout rate. Defaults to 0.1.
        qkv_bias (bool): Whether to use bias. Defaults to False.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """

    def __init__(self,
                 d_model: int = 512,
                 d_inner: int = 512,
                 n_head: int = 8,
                 d_k: int = 64,
                 d_v: int = 64,
                 dropout: float = 0.1,
                 qkv_bias: bool = False,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, qkv_bias=qkv_bias, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = LocalityAwareFeedforward(d_model, d_inner)

    def forward(self,
                x: Tensor,
                h: int,
                w: int,
                mask: Optional[Tensor] = None) -> Tensor:
        """Forward propagation of encoder.

        Args:
            x (Tensor): Feature tensor of shape :math:`(N, h*w, D_m)`.
            h (int): Height of the original feature.
            w (int): Width of the original feature.
            mask (Tensor, optional): Mask used for masked multi-head attention.
                Defaults to None.

        Returns:
            Tensor: A tensor of shape :math:`(N, h*w, D_m)`.
        """
        n, hw, c = x.size()
        residual = x
        x = self.norm1(x)
        x = residual + self.attn(x, x, x, mask)
        residual = x
        x = self.norm2(x)
        x = x.transpose(1, 2).contiguous().view(n, c, h, w)
        x = self.feed_forward(x)
        x = x.view(n, c, hw).transpose(1, 2)
        x = residual + x
        return x


class LocalityAwareFeedforward(BaseModule):
    """Locality-aware feedforward layer in SATRN, see `SATRN.

    <https://arxiv.org/abs/1910.04396>`_

    Args:
        d_in (int): Dimension of the input features.
        d_hid (int): Hidden dimension of feedforward layers.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to [dict(type='Xavier', layer='Conv2d'),
            dict(type='Constant', layer='BatchNorm2d', val=1, bias=0)].
    """

    def __init__(
        self,
        d_in: int,
        d_hid: int,
        init_cfg: Optional[Union[Dict, List[Dict]]] = [
            dict(type='Xavier', layer='Conv2d'),
            dict(type='Constant', layer='BatchNorm2d', val=1, bias=0)
        ]
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.conv1 = ConvModule(
            d_in,
            d_hid,
            kernel_size=1,
            padding=0,
            bias=False,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))

        self.depthwise_conv = ConvModule(
            d_hid,
            d_hid,
            kernel_size=3,
            padding=1,
            bias=False,
            groups=d_hid,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))

        self.conv2 = ConvModule(
            d_hid,
            d_in,
            kernel_size=1,
            padding=0,
            bias=False,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))

    def forward(self, x: Tensor) -> Tensor:
        """Forward propagation of Locality Aware Feedforward module.

        Args:
            x (Tensor): Feature tensor.

        Returns:
            Tensor: Feature tensor after Locality Aware Feedforward.
        """
        x = self.conv1(x)
        x = self.depthwise_conv(x)
        x = self.conv2(x)
        return x


class Adaptive2DPositionalEncoding(BaseModule):
    """Implement Adaptive 2D positional encoder for SATRN, see `SATRN.

    <https://arxiv.org/abs/1910.04396>`_ Modified from
    https://github.com/Media-Smart/vedastr Licensed under the Apache License,
    Version 2.0 (the "License");

    Args:
        d_hid (int): Dimensions of hidden layer. Defaults to 512.
        n_height (int): Max height of the 2D feature output. Defaults to 100.
        n_width (int): Max width of the 2D feature output. Defaults to 100.
        dropout (float): Dropout rate. Defaults to 0.1.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to [dict(type='Xavier', layer='Conv2d')]
    """

    def __init__(
        self,
        d_hid: int = 512,
        n_height: int = 100,
        n_width: int = 100,
        dropout: float = 0.1,
        init_cfg: Optional[Union[Dict, List[Dict]]] = [
            dict(type='Xavier', layer='Conv2d')
        ]
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        h_position_encoder = self._get_sinusoid_encoding_table(n_height, d_hid)
        h_position_encoder = h_position_encoder.transpose(0, 1)
        h_position_encoder = h_position_encoder.view(1, d_hid, n_height, 1)

        w_position_encoder = self._get_sinusoid_encoding_table(n_width, d_hid)
        w_position_encoder = w_position_encoder.transpose(0, 1)
        w_position_encoder = w_position_encoder.view(1, d_hid, 1, n_width)

        self.register_buffer('h_position_encoder', h_position_encoder)
        self.register_buffer('w_position_encoder', w_position_encoder)

        self.h_scale = self._scale_factor_generate(d_hid)
        self.w_scale = self._scale_factor_generate(d_hid)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)

    @staticmethod
    def _get_sinusoid_encoding_table(n_position: int, d_hid: int) -> Tensor:
        """Generate sinusoid position encoding table."""
        denominator = torch.Tensor([
            1.0 / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ])
        denominator = denominator.view(1, -1)
        pos_tensor = torch.arange(n_position).unsqueeze(-1).float()
        sinusoid_table = pos_tensor * denominator
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])

        return sinusoid_table

    @staticmethod
    def _scale_factor_generate(d_hid: int) -> nn.Sequential:
        """Generate scale factor layers."""
        scale_factor = nn.Sequential(
            nn.Conv2d(d_hid, d_hid, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(d_hid, d_hid, kernel_size=1), nn.Sigmoid())

        return scale_factor

    def forward(self, x: Tensor) -> Tensor:
        """Forward propagation of Locality Aware Feedforward module.

        Args:
            x (Tensor): Feature tensor.

        Returns:
            Tensor: Feature tensor after Locality Aware Feedforward.
        """
        _, _, h, w = x.size()
        avg_pool = self.pool(x)
        h_pos_encoding = \
            self.h_scale(avg_pool) * self.h_position_encoder[:, :, :h, :]
        w_pos_encoding = \
            self.w_scale(avg_pool) * self.w_position_encoder[:, :, :, :w]
        out = x + h_pos_encoding + w_pos_encoding
        out = self.dropout(out)

        return out

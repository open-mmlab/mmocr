# Copyright (c) OpenMMLab. All rights reserved.
"""This code is from https://github.com/jadore801120/attention-is-all-you-need-
pytorch."""
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule


class TransformerEncoderLayer(nn.Module):
    """"""

    def __init__(self,
                 d_model=512,
                 d_inner=256,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 qkv_bias=False,
                 mask_value=0,
                 act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(
            n_head,
            d_model,
            d_k,
            d_v,
            qkv_bias=qkv_bias,
            dropout=dropout,
            mask_value=mask_value)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, act_layer=act_layer)

    def forward(self, x, mask=None):
        residual = x
        x = self.norm1(x)
        x = residual + self.attn(x, x, x, mask)
        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)

        return x


class SatrnEncoderLayer(BaseModule):
    """"""

    def __init__(self,
                 d_model=512,
                 d_inner=512,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 qkv_bias=False,
                 mask_value=0,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(
            n_head,
            d_model,
            d_k,
            d_v,
            qkv_bias=qkv_bias,
            dropout=dropout,
            mask_value=mask_value)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = LocalityAwareFeedforward(
            d_model, d_inner, dropout=dropout)

    def forward(self, x, h, w, mask=None):
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


class TransformerDecoderLayer(nn.Module):

    def __init__(self,
                 d_model=512,
                 d_inner=256,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 qkv_bias=False,
                 mask_value=0,
                 act_layer=nn.GELU):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.self_attn = MultiHeadAttention(
            n_head,
            d_model,
            d_k,
            d_v,
            dropout=dropout,
            qkv_bias=qkv_bias,
            mask_value=mask_value)
        self.enc_attn = MultiHeadAttention(
            n_head,
            d_model,
            d_k,
            d_v,
            dropout=dropout,
            qkv_bias=qkv_bias,
            mask_value=mask_value)
        self.mlp = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, act_layer=act_layer)

    def forward(self,
                dec_input,
                enc_output,
                self_attn_mask=None,
                dec_enc_attn_mask=None):
        self_attn_in = self.norm1(dec_input)
        self_attn_out = self.self_attn(self_attn_in, self_attn_in,
                                       self_attn_in, self_attn_mask)
        enc_attn_in = dec_input + self_attn_out

        enc_attn_q = self.norm2(enc_attn_in)
        enc_attn_out = self.enc_attn(enc_attn_q, enc_output, enc_output,
                                     dec_enc_attn_mask)

        mlp_in = enc_attn_in + enc_attn_out
        mlp_out = self.mlp(self.norm3(mlp_in))
        out = mlp_in + mlp_out

        return out


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module."""

    def __init__(self,
                 n_head=8,
                 d_model=512,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 qkv_bias=False,
                 mask_value=0):
        super().__init__()

        self.mask_value = mask_value

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.scale = d_k**-0.5

        self.dim_k = n_head * d_k
        self.dim_v = n_head * d_v

        self.linear_q = nn.Linear(self.dim_k, self.dim_k, bias=qkv_bias)

        self.linear_k = nn.Linear(self.dim_k, self.dim_k, bias=qkv_bias)

        self.linear_v = nn.Linear(self.dim_v, self.dim_v, bias=qkv_bias)

        self.fc = nn.Linear(self.dim_v, d_model, bias=qkv_bias)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size, len_q, _ = q.size()
        _, len_k, _ = k.size()

        q = self.linear_q(q).view(batch_size, len_q, self.n_head, self.d_k)
        k = self.linear_k(k).view(batch_size, len_k, self.n_head, self.d_k)
        v = self.linear_v(v).view(batch_size, len_k, self.n_head, self.d_v)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 1, 3)

        logits = torch.matmul(q, k) * self.scale

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            elif mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            logits = logits.masked_fill(mask == self.mask_value, float('-inf'))
        weights = logits.softmax(dim=-1)
        weights = self.attn_drop(weights)

        attn_out = torch.matmul(weights, v).transpose(1, 2)
        attn_out = attn_out.reshape(batch_size, len_q, self.dim_v)
        attn_out = self.fc(attn_out)
        attn_out = self.proj_drop(attn_out)

        return attn_out


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module."""

    def __init__(self, d_in, d_hid, dropout=0.1, act_layer=nn.GELU):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.act = act_layer()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)

        return x


class LocalityAwareFeedforward(BaseModule):
    """Locality-aware feedforward layer in SATRN, see `SATRN.

    <https://arxiv.org/abs/1910.04396>`_
    """

    def __init__(self,
                 d_in,
                 d_hid,
                 dropout=0.1,
                 init_cfg=[
                     dict(type='Xavier', layer='Conv2d'),
                     dict(type='Constant', layer='BatchNorm2d', val=1, bias=0)
                 ]):
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

    def forward(self, x):
        x = self.conv1(x)
        x = self.depthwise_conv(x)
        x = self.conv2(x)

        return x


class PositionAttention(nn.Module):
    """Position attention module that transcribes visual features into
    character attention module. Serves as a part of implementation of ABINet
    (https://arxiv.org/abs/1910.04396).

    Adapted from https://github.com/FangShancheng/ABINet.
    """

    def __init__(self,
                 max_length,
                 in_channels=512,
                 num_channels=64,
                 h=8,
                 w=32,
                 mode='nearest',
                 **kwargs):
        super().__init__()
        self.max_length = max_length

        self.k_encoder = nn.Sequential(
            self._encoder_layer(in_channels, num_channels, stride=(1, 2)),
            self._encoder_layer(num_channels, num_channels, stride=(2, 2)),
            self._encoder_layer(num_channels, num_channels, stride=(2, 2)),
            self._encoder_layer(num_channels, num_channels, stride=(2, 2)))

        self.k_decoder = nn.Sequential(
            self._decoder_layer(
                num_channels, num_channels, scale_factor=2, mode=mode),
            self._decoder_layer(
                num_channels, num_channels, scale_factor=2, mode=mode),
            self._decoder_layer(
                num_channels, num_channels, scale_factor=2, mode=mode),
            self._decoder_layer(
                num_channels, in_channels, size=(h, w), mode=mode))

        self.pos_encoder = PositionalEncoding(in_channels, max_length)
        self.project = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        N, E, H, W = x.size()
        k, v = x, x  # (N, E, H, W)

        # Apply mini U-Net on k
        features = []
        for i in range(len(self.k_encoder)):
            k = self.k_encoder[i](k)
            features.append(k)
        for i in range(len(self.k_decoder) - 1):
            k = self.k_decoder[i](k)
            k = k + features[len(self.k_decoder) - 2 - i]
        k = self.k_decoder[-1](k)

        # q = positional encoding
        zeros = x.new_zeros((self.max_length, N, E))  # (T, N, E)
        q = self.pos_encoder(zeros)  # (T, N, E)
        q = q.permute(1, 0, 2)  # (N, T, E)
        q = self.project(q)  # (N, T, E)

        # Attention encoding
        attn_scores = torch.bmm(q, k.flatten(2, 3))  # (N, T, (H*W))
        attn_scores = attn_scores / (E**0.5)
        attn_scores = torch.softmax(attn_scores, dim=-1)

        v = v.permute(0, 2, 3, 1).view(N, -1, E)  # (N, (H*W), E)
        attn_vecs = torch.bmm(attn_scores, v)  # (N, T, E)

        return attn_vecs, attn_scores.view(N, -1, H, W)

    def _encoder_layer(self,
                       in_channels,
                       out_channels,
                       kernel_size=3,
                       stride=2,
                       padding=1):
        return ConvModule(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))

    def _decoder_layer(self,
                       in_channels,
                       out_channels,
                       kernel_size=3,
                       stride=1,
                       padding=1,
                       mode='nearest',
                       scale_factor=None,
                       size=None):
        align_corners = None if mode == 'nearest' else True
        return nn.Sequential(
            nn.Upsample(
                size=size,
                scale_factor=scale_factor,
                mode=mode,
                align_corners=align_corners),
            ConvModule(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU')))


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid=512, n_position=200, dropout=0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Not a parameter
        self.register_buffer(
            'position_table',
            self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table."""
        denominator = torch.Tensor([
            1.0 / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ])
        denominator = denominator.view(1, -1)
        pos_tensor = torch.arange(n_position).unsqueeze(-1).float()
        sinusoid_table = pos_tensor * denominator
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])

        return sinusoid_table.unsqueeze(0)

    def forward(self, x):
        self.device = x.device
        x = x + self.position_table[:, :x.size(1)].clone().detach()
        return self.dropout(x)


class Adaptive2DPositionalEncoding(BaseModule):
    """Implement Adaptive 2D positional encoder for SATRN, see
      `SATRN <https://arxiv.org/abs/1910.04396>`_
      Modified from https://github.com/Media-Smart/vedastr
      Licensed under the Apache License, Version 2.0 (the "License");
    Args:
        d_hid (int): Dimensions of hidden layer.
        n_height (int): Max height of the 2D feature output.
        n_width (int): Max width of the 2D feature output.
        dropout (int): Size of hidden layers of the model.
    """

    def __init__(self,
                 d_hid=512,
                 n_height=100,
                 n_width=100,
                 dropout=0.1,
                 init_cfg=[dict(type='Xavier', layer='Conv2d')]):
        super().__init__(init_cfg=init_cfg)

        h_position_encoder = self._get_sinusoid_encoding_table(n_height, d_hid)
        h_position_encoder = h_position_encoder.transpose(0, 1)
        h_position_encoder = h_position_encoder.view(1, d_hid, n_height, 1)

        w_position_encoder = self._get_sinusoid_encoding_table(n_width, d_hid)
        w_position_encoder = w_position_encoder.transpose(0, 1)
        w_position_encoder = w_position_encoder.view(1, d_hid, 1, n_width)

        self.register_buffer('h_position_encoder', h_position_encoder)
        self.register_buffer('w_position_encoder', w_position_encoder)

        self.h_scale = self.scale_factor_generate(d_hid)
        self.w_scale = self.scale_factor_generate(d_hid)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table."""
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

    def scale_factor_generate(self, d_hid):
        scale_factor = nn.Sequential(
            nn.Conv2d(d_hid, d_hid, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(d_hid, d_hid, kernel_size=1), nn.Sigmoid())

        return scale_factor

    def forward(self, x):
        b, c, h, w = x.size()

        avg_pool = self.pool(x)

        h_pos_encoding = \
            self.h_scale(avg_pool) * self.h_position_encoder[:, :, :h, :]
        w_pos_encoding = \
            self.w_scale(avg_pool) * self.w_position_encoder[:, :, :, :w]

        out = x + h_pos_encoding + w_pos_encoding

        out = self.dropout(out)

        return out


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    """For masking out the subsequent info."""
    len_s = seq.size(1)
    subsequent_mask = 1 - torch.triu(
        torch.ones((len_s, len_s), device=seq.device), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).bool()
    return subsequent_mask

# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.runner import BaseModule

from mmocr.models.common.modules import (MultiHeadAttention,
                                         PositionwiseFeedForward)


class TFEncoderLayer(BaseModule):
    """Transformer Encoder Layer."""

    def __init__(self,
                 d_model=512,
                 d_inner=256,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 qkv_bias=False,
                 act_cfg=dict(type='GELU')):
        super().__init__()
        self.attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, qkv_bias=qkv_bias, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, act_cfg=act_cfg)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        residual = x
        x = residual + self.attn(x, x, x, mask)
        x = self.norm1(x)

        residual = x
        x = residual + self.mlp(x)
        x = self.norm2(x)

        return x


class TFDecoderLayer(nn.Module):
    """Transformer Decoder Layer."""

    def __init__(self,
                 d_model=512,
                 d_inner=256,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 qkv_bias=False,
                 act_cfg=dict(type='GELU')):
        super().__init__()
        self.self_attn = MultiHeadAttention()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.self_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, qkv_bias=qkv_bias)

        self.enc_dec_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, qkv_bias=qkv_bias)

        self.mlp = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, act_cfg=act_cfg)

    def forward(self,
                dec_input,
                enc_output,
                self_attn_mask=None,
                dec_enc_attn_mask=None):

        dec_attn_out = self.self_attn(dec_input, dec_input, dec_input,
                                      self_attn_mask)
        dec_attn_out += dec_input
        dec_attn_out = self.norm1(dec_attn_out)

        enc_dec_attn_out = self.enc_dec_attn(dec_attn_out, enc_output,
                                             enc_output, dec_enc_attn_mask)
        enc_dec_attn_out += dec_attn_out
        enc_dec_attn_out = self.norm2(enc_dec_attn_out)

        mlp_out = self.mlp(enc_dec_attn_out)
        mlp_out += enc_dec_attn_out
        out = self.norm3(mlp_out)

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

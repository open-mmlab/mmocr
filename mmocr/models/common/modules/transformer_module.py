import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention Module. This code is adopted from
    https://github.com/jadore801120/attention-is-all-you-need-pytorch.

    Args:
        temperature (float): The scale factor for softmax input.
        attn_dropout (float): Dropout probability.
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module.

    Args:
        n_head (int): The number of heads in the
            multi-head attention models (default=8).
        d_model (int): The number of expected features in the
            encoder/decoder inputs (default=512).
        d_k (int): The dimension of key in attention block.
        d_v (int): The dimension of value in attention block.
    """

    def __init__(self,
                 n_head=8,
                 d_model=512,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 qkv_bias=False):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.dim_k = n_head * d_k
        self.dim_v = n_head * d_v

        self.linear_q = nn.Linear(self.dim_k, self.dim_k, bias=qkv_bias)
        self.linear_k = nn.Linear(self.dim_k, self.dim_k, bias=qkv_bias)
        self.linear_v = nn.Linear(self.dim_v, self.dim_v, bias=qkv_bias)

        self.attention = ScaledDotProductAttention(d_k**0.5, dropout)

        self.fc = nn.Linear(self.dim_v, d_model, bias=qkv_bias)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size, len_q, _ = q.size()
        _, len_k, _ = k.size()

        q = self.linear_q(q).view(batch_size, len_q, self.n_head, self.d_k)
        k = self.linear_k(k).view(batch_size, len_k, self.n_head, self.d_k)
        v = self.linear_v(v).view(batch_size, len_k, self.n_head, self.d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            elif mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)

        attn_out, _ = self.attention(q, k, v, mask=mask)

        attn_out = attn_out.transpose(1, 2).contiguous().view(
            batch_size, len_q, self.dim_v)

        attn_out = self.fc(attn_out)
        attn_out = self.proj_drop(attn_out)

        return attn_out


class PositionwiseFeedForward(nn.Module):
    """Two-layer feed-forward module."""

    def __init__(self, d_in, d_hid, dropout=0.1, act_cfg=dict(type='Relu')):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.act = build_activation_layer(act_cfg)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = self.act(x)
        x = self.w_2(x)
        x = self.dropout(x)

        return x


class PositionalEncoding(nn.Module):
    """Fixed positional encoding with sine and cosine functions."""

    def __init__(self, d_hid=512, n_position=200):
        super().__init__()
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
        return x + self.position_table[:, :x.size(1)].clone().detach()

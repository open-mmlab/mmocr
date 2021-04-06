"""This code is from https://github.com/jadore801120/attention-is-all-you-need-
pytorch."""
import numpy as np
import torch
import torch.nn as nn


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
        self.self_attn = MultiHeadAttention()

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


class PositionalEncoding(nn.Module):

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


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    """For masking out the subsequent info."""
    len_s = seq.size(1)
    subsequent_mask = 1 - torch.triu(
        torch.ones((len_s, len_s), device=seq.device), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).bool()
    return subsequent_mask

"""This code is from https://github.com/jadore801120/attention-is-all-you-need-
pytorch."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderLayer(nn.Module):
    """Compose with three layers：
        1. MultiHeadSelfAttn
        2. MultiHeadEncoderDecoderAttn
        3. PositionwiseFeedForward
    """

    def __init__(self,
                 d_model=512,
                 d_inner=256,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout)

    def forward(self,
                dec_input,
                enc_output,
                slf_attn_mask=None,
                dec_enc_attn_mask=None):

        dec_output = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)

        return dec_output


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention."""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(1, 2))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module."""

    def __init__(self, n_head=8, d_model=512, d_k=64, d_v=64, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs_list = nn.ModuleList(
            [nn.Linear(d_model, d_k, bias=False) for _ in range(n_head)])
        self.w_ks_list = nn.ModuleList(
            [nn.Linear(d_model, d_k, bias=False) for _ in range(n_head)])
        self.w_vs_list = nn.ModuleList(
            [nn.Linear(d_model, d_v, bias=False) for _ in range(n_head)])
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        residual = q
        q = self.layer_norm(q)

        attention_q_list = []
        for head_index in range(self.n_head):
            q_each = self.w_qs_list[head_index](q)  # bsz * seq_len * d_k
            k_each = self.w_ks_list[head_index](k)  # bsz * seq_len * d_k
            v_each = self.w_vs_list[head_index](v)  # bsz * seq_len * d_v
            attention_q_each, _ = self.attention(
                q_each, k_each, v_each, mask=mask)
            attention_q_list.append(attention_q_each)

        q = torch.cat(attention_q_list, dim=-1)
        q = self.dropout(self.fc(q))
        q += residual

        return q


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module."""

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x
        x = self.layer_norm(x)

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

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

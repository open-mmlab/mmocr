# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import BaseTransformerLayer
from mmcv.runner import ModuleList

from mmocr.models.common.modules import PositionalEncoding
from mmocr.registry import MODELS
from .base_decoder import BaseDecoder


@MODELS.register_module()
class ABILanguageDecoder(BaseDecoder):
    r"""Transformer-based language model responsible for spell correction.
    Implementation of language model of \
        `ABINet <https://arxiv.org/abs/1910.04396>`_.

    Args:
        d_model (int): Hidden size of input.
        n_head (int): Number of multi-attention heads.
        d_inner (int): Hidden size of feedforward network model.
        n_layers (int): The number of similar decoding layers.
        max_seq_len (int): Maximum text sequence length :math:`T`.
        dropout (float): Dropout rate.
        detach_tokens (bool): Whether to block the gradient flow at input
         tokens.
        num_chars (int): Number of text characters :math:`C`.
        use_self_attn (bool): If True, use self attention in decoder layers,
            otherwise cross attention will be used.
        pad_idx (bool): The index of the token indicating the end of output,
            which is used to compute the length of output. It is usually the
            index of `<EOS>` or `<PAD>` token.
        init_cfg (dict): Specifies the initialization method for model layers.
    """

    def __init__(self,
                 d_model=512,
                 n_head=8,
                 d_inner=2048,
                 n_layers=4,
                 max_seq_len=40,
                 dropout=0.1,
                 detach_tokens=True,
                 num_chars=90,
                 use_self_attn=False,
                 pad_idx=0,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.detach_tokens = detach_tokens

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.proj = nn.Linear(num_chars, d_model, False)
        self.token_encoder = PositionalEncoding(
            d_model, n_position=self.max_seq_len, dropout=0.1)
        self.pos_encoder = PositionalEncoding(
            d_model, n_position=self.max_seq_len)
        self.pad_idx = pad_idx

        if use_self_attn:
            operation_order = ('self_attn', 'norm', 'cross_attn', 'norm',
                               'ffn', 'norm')
        else:
            operation_order = ('cross_attn', 'norm', 'ffn', 'norm')

        decoder_layer = BaseTransformerLayer(
            operation_order=operation_order,
            attn_cfgs=dict(
                type='MultiheadAttention',
                embed_dims=d_model,
                num_heads=n_head,
                attn_drop=dropout,
                dropout_layer=dict(type='Dropout', drop_prob=dropout),
            ),
            ffn_cfgs=dict(
                type='FFN',
                embed_dims=d_model,
                feedforward_channels=d_inner,
                ffn_drop=dropout,
            ),
            norm_cfg=dict(type='LN'),
        )
        self.decoder_layers = ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(n_layers)])

        self.cls = nn.Linear(d_model, num_chars)

    def forward_train(self, feat, logits, targets_dict, img_metas):
        """
        Args:
            logits (Tensor): Raw language logitis. Shape (N, T, C).

        Returns:
            A dict with keys ``feature`` and ``logits``.
            feature (Tensor): Shape (N, T, E). Raw textual features for vision
                language aligner.
            logits (Tensor): Shape (N, T, C). The raw logits for characters
                after spell correction.
        """
        lengths = self._get_length(logits)
        lengths.clamp_(2, self.max_seq_len)
        tokens = torch.softmax(logits, dim=-1)
        if self.detach_tokens:
            tokens = tokens.detach()
        embed = self.proj(tokens)  # (N, T, E)
        embed = self.token_encoder(embed)  # (N, T, E)
        padding_mask = self._get_padding_mask(lengths, self.max_seq_len)

        zeros = embed.new_zeros(*embed.shape)
        query = self.pos_encoder(zeros)
        query = query.permute(1, 0, 2)  # (T, N, E)
        embed = embed.permute(1, 0, 2)
        location_mask = self._get_location_mask(self.max_seq_len,
                                                tokens.device)
        output = query
        for m in self.decoder_layers:
            output = m(
                query=output,
                key=embed,
                value=embed,
                attn_masks=location_mask,
                key_padding_mask=padding_mask)
        output = output.permute(1, 0, 2)  # (N, T, E)

        logits = self.cls(output)  # (N, T, C)
        return {'feature': output, 'logits': logits}

    def forward_test(self, feat, out_enc, img_metas):
        return self.forward_train(feat, out_enc, None, img_metas)

    def _get_length(self, logit, dim=-1):
        """Greedy decoder to obtain length from logit.

        Returns the first location of padding index or the length of the entire
        tensor otherwise.
        """
        # out as a boolean vector indicating the existence of end token(s)
        out = (logit.argmax(dim=-1) == self.pad_idx)
        abn = out.any(dim)
        # Get the first index of end token
        out = ((out.cumsum(dim) == 1) & out).max(dim)[1]
        out = out + 1
        out = torch.where(abn, out, out.new_tensor(logit.shape[1]))
        return out

    @staticmethod
    def _get_location_mask(seq_len, device=None):
        """Generate location masks given input sequence length.

        Args:
            seq_len (int): The length of input sequence to transformer.
            device (torch.device or str, optional): The device on which the
                masks will be placed.

        Returns:
            Tensor: A mask tensor of shape (seq_len, seq_len) with -infs on
            diagonal and zeros elsewhere.
        """
        mask = torch.eye(seq_len, device=device)
        mask = mask.float().masked_fill(mask == 1, float('-inf'))
        return mask

    @staticmethod
    def _get_padding_mask(length, max_length):
        """Generate padding masks.

        Args:
            length (Tensor): Shape :math:`(N,)`.
            max_length (int): The maximum sequence length :math:`T`.

        Returns:
            Tensor: A bool tensor of shape :math:`(N, T)` with Trues on
            elements located over the length, or Falses elsewhere.
        """
        length = length.unsqueeze(-1)
        grid = torch.arange(0, max_length, device=length.device).unsqueeze(0)
        return grid >= length

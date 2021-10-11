# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import BaseTransformerLayer
from mmcv.runner import ModuleList

from mmocr.models.builder import DECODERS
from mmocr.models.textrecog.layers import PositionalEncoding
from .base_decoder import BaseDecoder


@DECODERS.register_module()
class ABIVisionDecoder(BaseDecoder):

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
        lengths = self._get_length(logits)
        lengths.clamp_(2, self.max_seq_len)
        tokens = torch.softmax(logits, dim=-1)
        if self.detach_tokens:
            tokens = tokens.detach()
        embed = self.proj(tokens)  # (N, T, E)
        embed = self.token_encoder(embed)  # (T, N, E)
        padding_mask = self._get_padding_mask(lengths, self.max_seq_len)

        zeros = embed.new_zeros(*embed.shape)
        query = self.pos_encoder(zeros)
        query = query.permute(1, 0, 2)
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

    def forward(self,
                feat,
                out_enc,
                targets_dict=None,
                img_metas=None,
                train_mode=True):
        self.train_mode = train_mode
        if train_mode:
            return self.forward_train(feat, out_enc, targets_dict, img_metas)

        return self.forward_test(feat, out_enc, img_metas)

    def _get_length(self, logit, dim=-1):
        """Greed decoder to obtain length from logit Returns the first location
        of padding index or the length of the entire tensor otherwise."""
        # out as a boolean vector indicating the existence of end token(s)
        out = (logit.argmax(dim=-1) == self.pad_idx)
        abn = out.any(dim)
        # get the first index of end token
        out = ((out.cumsum(dim) == 1) & out).max(dim)[1]
        out = out + 1
        out = torch.where(abn, out, out.new_tensor(logit.shape[1]))
        return out

    @staticmethod
    def _get_location_mask(sz, device=None):
        mask = torch.eye(sz, device=device)
        mask = mask.float().masked_fill(mask == 1, float('-inf'))
        return mask

    @staticmethod
    def _get_padding_mask(length, max_length):
        length = length.unsqueeze(-1)
        grid = torch.arange(0, max_length, device=length.device).unsqueeze(0)
        return grid >= length

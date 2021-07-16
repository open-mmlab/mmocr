import math

import torch
import torch.nn as nn

from mmocr.models.builder import DECODERS
from mmocr.models.textrecog.layers import (DotProductAttentionLayer,
                                           PositionAwareLayer)
from .base_decoder import BaseDecoder


@DECODERS.register_module()
class PositionAttentionDecoder(BaseDecoder):

    def __init__(self,
                 num_classes=None,
                 rnn_layers=2,
                 dim_input=512,
                 dim_model=128,
                 max_seq_len=40,
                 mask=True,
                 return_feature=False,
                 encode_value=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.num_classes = num_classes
        self.dim_input = dim_input
        self.dim_model = dim_model
        self.max_seq_len = max_seq_len
        self.return_feature = return_feature
        self.encode_value = encode_value
        self.mask = mask

        self.embedding = nn.Embedding(self.max_seq_len + 1, self.dim_model)

        self.position_aware_module = PositionAwareLayer(
            self.dim_model, rnn_layers)

        self.attention_layer = DotProductAttentionLayer()

        self.prediction = None
        if not self.return_feature:
            pred_num_classes = num_classes - 1
            self.prediction = nn.Linear(
                dim_model if encode_value else dim_input, pred_num_classes)

    def _get_position_index(self, length, batch_size, device=None):
        position_index = torch.arange(0, length, device=device)
        position_index = position_index.repeat([batch_size, 1])
        position_index = position_index.long()
        return position_index

    def forward_train(self, feat, out_enc, targets_dict, img_metas):
        valid_ratios = [
            img_meta.get('valid_ratio', 1.0) for img_meta in img_metas
        ] if self.mask else None

        targets = targets_dict['padded_targets'].to(feat.device)

        #
        n, c_enc, h, w = out_enc.size()
        assert c_enc == self.dim_model
        _, c_feat, _, _ = feat.size()
        assert c_feat == self.dim_input
        _, len_q = targets.size()
        assert len_q <= self.max_seq_len

        position_index = self._get_position_index(len_q, n, feat.device)

        position_out_enc = self.position_aware_module(out_enc)

        query = self.embedding(position_index)
        query = query.permute(0, 2, 1).contiguous()
        key = position_out_enc.view(n, c_enc, h * w)
        if self.encode_value:
            value = out_enc.view(n, c_enc, h * w)
        else:
            value = feat.view(n, c_feat, h * w)

        mask = None
        if valid_ratios is not None:
            mask = query.new_zeros((n, h, w))
            for i, valid_ratio in enumerate(valid_ratios):
                valid_width = min(w, math.ceil(w * valid_ratio))
                mask[i, :, valid_width:] = 1
            mask = mask.bool()
            mask = mask.view(n, h * w)

        attn_out = self.attention_layer(query, key, value, mask)
        attn_out = attn_out.permute(0, 2, 1).contiguous()  # [n, len_q, dim_v]

        if self.return_feature:
            return attn_out

        return self.prediction(attn_out)

    def forward_test(self, feat, out_enc, img_metas):
        valid_ratios = [
            img_meta.get('valid_ratio', 1.0) for img_meta in img_metas
        ] if self.mask else None

        seq_len = self.max_seq_len
        n, c_enc, h, w = out_enc.size()
        assert c_enc == self.dim_model
        _, c_feat, _, _ = feat.size()
        assert c_feat == self.dim_input

        position_index = self._get_position_index(seq_len, n, feat.device)

        position_out_enc = self.position_aware_module(out_enc)

        query = self.embedding(position_index)
        query = query.permute(0, 2, 1).contiguous()
        key = position_out_enc.view(n, c_enc, h * w)
        if self.encode_value:
            value = out_enc.view(n, c_enc, h * w)
        else:
            value = feat.view(n, c_feat, h * w)

        mask = None
        if valid_ratios is not None:
            mask = query.new_zeros((n, h, w))
            for i, valid_ratio in enumerate(valid_ratios):
                valid_width = min(w, math.ceil(w * valid_ratio))
                mask[i, :, valid_width:] = 1
            mask = mask.bool()
            mask = mask.view(n, h * w)

        attn_out = self.attention_layer(query, key, value, mask)
        attn_out = attn_out.permute(0, 2, 1).contiguous()

        if self.return_feature:
            return attn_out

        return self.prediction(attn_out)

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmocr.models.builder import DECODERS
from mmocr.models.textrecog.layers import DotProductAttentionLayer
from .base_decoder import BaseDecoder


@DECODERS.register_module()
class SequenceAttentionDecoder(BaseDecoder):

    def __init__(self,
                 num_classes=None,
                 rnn_layers=2,
                 dim_input=512,
                 dim_model=128,
                 max_seq_len=40,
                 start_idx=0,
                 mask=True,
                 padding_idx=None,
                 dropout_ratio=0,
                 return_feature=False,
                 encode_value=False):
        super().__init__()

        self.num_classes = num_classes
        self.dim_input = dim_input
        self.dim_model = dim_model
        self.return_feature = return_feature
        self.encode_value = encode_value
        self.max_seq_len = max_seq_len
        self.start_idx = start_idx
        self.mask = mask

        self.embedding = nn.Embedding(
            self.num_classes, self.dim_model, padding_idx=padding_idx)

        self.sequence_layer = nn.LSTM(
            input_size=dim_model,
            hidden_size=dim_model,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout_ratio)

        self.attention_layer = DotProductAttentionLayer()

        self.prediction = None
        if not self.return_feature:
            pred_num_classes = num_classes - 1
            self.prediction = nn.Linear(
                dim_model if encode_value else dim_input, pred_num_classes)

    def init_weights(self):
        pass

    def forward_train(self, feat, out_enc, targets_dict, img_metas):
        valid_ratios = [
            img_meta.get('valid_ratio', 1.0) for img_meta in img_metas
        ] if self.mask else None

        targets = targets_dict['padded_targets'].to(feat.device)
        tgt_embedding = self.embedding(targets)

        n, c_enc, h, w = out_enc.size()
        assert c_enc == self.dim_model
        _, c_feat, _, _ = feat.size()
        assert c_feat == self.dim_input
        _, len_q, c_q = tgt_embedding.size()
        assert c_q == self.dim_model
        assert len_q <= self.max_seq_len

        query, _ = self.sequence_layer(tgt_embedding)
        query = query.permute(0, 2, 1).contiguous()
        key = out_enc.view(n, c_enc, h * w)
        if self.encode_value:
            value = key
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

        out = self.prediction(attn_out)

        return out

    def forward_test(self, feat, out_enc, img_metas):
        seq_len = self.max_seq_len
        batch_size = feat.size(0)

        decode_sequence = (feat.new_ones(
            (batch_size, seq_len)) * self.start_idx).long()

        outputs = []
        for i in range(seq_len):
            step_out = self.forward_test_step(feat, out_enc, decode_sequence,
                                              i, img_metas)
            outputs.append(step_out)
            _, max_idx = torch.max(step_out, dim=1, keepdim=False)
            if i < seq_len - 1:
                decode_sequence[:, i + 1] = max_idx

        outputs = torch.stack(outputs, 1)

        return outputs

    def forward_test_step(self, feat, out_enc, decode_sequence, current_step,
                          img_metas):
        valid_ratios = [
            img_meta.get('valid_ratio', 1.0) for img_meta in img_metas
        ] if self.mask else None

        embed = self.embedding(decode_sequence)

        n, c_enc, h, w = out_enc.size()
        assert c_enc == self.dim_model
        _, c_feat, _, _ = feat.size()
        assert c_feat == self.dim_input
        _, _, c_q = embed.size()
        assert c_q == self.dim_model

        query, _ = self.sequence_layer(embed)
        query = query.permute(0, 2, 1).contiguous()
        key = out_enc.view(n, c_enc, h * w)
        if self.encode_value:
            value = key
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

        # [n, c, l]
        attn_out = self.attention_layer(query, key, value, mask)

        out = attn_out[:, :, current_step]

        if self.return_feature:
            return out

        out = self.prediction(out)
        out = F.softmax(out, dim=-1)

        return out

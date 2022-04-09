# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmocr.models.builder import DECODERS
from mmocr.models.common.modules import PositionalEncoding
from .base_decoder import BaseDecoder

from mmcv.cnn.bricks.transformer import BaseTransformerLayer
from mmcv.runner import ModuleList


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Embeddings(nn.Module):

    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, *input):
        x = input[0]
        return self.lut(x) * math.sqrt(self.d_model)


@DECODERS.register_module()
class MasterDecoder(BaseDecoder):
    """Decoder module in `MASTER.

    <https://arxiv.org/abs/1910.02562>`_.

    Args:
        N (int): Number of transformer layer in decoder.
        decoder (dict): Transformer decoder setting config dict.
        d_model (int): Output channel of decoder.
        num_classes (int): Output class number.
        start_idx (int): Index of start token.
        padding_idx (int): Index of padding token.
        max_seq_len (int): Maximum sequence length for decoding.
    """

    def __init__(
        self,
        d_model,
        num_classes,
        start_idx,
        padding_idx,
        n_head=8,
        attn_drop=0.,
        ffn_drop=0.,
        dropout=0.,
        d_inner=2048,
        n_layers=3,
        max_seq_len=30,
        feat_pe_dropout=0.2,
        feat_size=6*40,
    ):
        super(MasterDecoder, self).__init__()

        operation_order = ('norm', 'self_attn', 'norm', 'cross_attn', 'norm', 'ffn')

        """
                    attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
        """
        decoder_layer = BaseTransformerLayer(
            operation_order=operation_order,
            attn_cfgs=dict(
                type='MultiheadAttention',
                embed_dims=d_model,
                num_heads=n_head,
                attn_drop=attn_drop,
                dropout_layer=dict(type='Dropout', drop_prob=dropout),
            ),
            ffn_cfgs=dict(
                type='FFN',
                embed_dims=d_model,
                feedforward_channels=d_inner,
                ffn_drop=ffn_drop,
                dropout_layer=dict(type='Dropout', drop_prob=dropout),
            ),
            norm_cfg=dict(type='LN'),
            batch_first=True,
        )
        self.decoder_layers = ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(n_layers)])

        self.cls = nn.Linear(d_model, num_classes)

        self.SOS = start_idx
        self.PAD = padding_idx
        self.max_seq_len = max_seq_len
        self.feat_size = feat_size
        self.n_head = n_head

        self.embedding = Embeddings(d_model=d_model, vocab=num_classes)
        self.positional_encoding = PositionalEncoding(
            d_hid=d_model, n_position=self.max_seq_len+1)
        self.feat_positional_encoding = PositionalEncoding(
            d_hid=d_model, n_position=self.feat_size, dropout=feat_pe_dropout)
        self.norm = nn.LayerNorm(d_model)

    def make_mask(self, src, tgt):
        """Make mask for self attention.

        :param src: [b, c, h, l_src]
        :param tgt: [b, l_tgt]
        :return:
        """
        trg_pad_mask = (tgt != self.PAD).unsqueeze(1).unsqueeze(3).byte()
        tgt_len = tgt.size(1)
        trg_sub_mask = torch.tril(
            torch.ones((tgt_len, tgt_len),
                       dtype=torch.uint8,
                       device=src.device))
        tgt_mask = trg_pad_mask & trg_sub_mask

        # inverse for mmcv's BaseTransformerLayer
        tril_mask = tgt_mask.clone()
        tgt_mask = tgt_mask.float().masked_fill_(tril_mask==0, -1e9)
        tgt_mask = tgt_mask.masked_fill_(tril_mask, 0)
        tgt_mask = tgt_mask.repeat(1, self.n_head, 1, 1)
        tgt_mask = tgt_mask.view(-1, tgt_len, tgt_len)
        return None, tgt_mask

    def decode(self, input, feature, src_mask, tgt_mask):
        x = self.embedding(input)
        x = self.positional_encoding(x)
        attn_masks = [tgt_mask, src_mask]
        for i, layer in enumerate(self.decoder_layers):
            x = layer(
                query=x,
                key=feature,
                value=feature,
                attn_masks=attn_masks)
        x = self.norm(x)
        return self.cls(x)

    def greedy_forward(self, SOS, feature):
        input = SOS
        output = None
        for i in range(self.max_seq_len + 1):
            _, target_mask = self.make_mask(feature, input)
            out = self.decode(input, feature, None, target_mask)
            output = out
            prob = F.softmax(out, dim=-1)
            _, next_word = torch.max(prob, dim=-1)
            input = torch.cat([input, next_word[:, -1].unsqueeze(-1)], dim=1)
        return output

    def forward_train(self, feat, out_enc, targets_dict, img_metas=None):
        # flatten 2D feature map
        if len(feat.shape) > 3:
            b, c, h, w = feat.shape
            feat = feat.view(b, c, h * w)
            feat = feat.permute((0, 2, 1))
        out_enc = self.feat_positional_encoding(feat) \
            if out_enc is None else out_enc

        device = feat.device
        if isinstance(targets_dict, dict):
            padded_targets = targets_dict['padded_targets'].to(device)
        else:
            padded_targets = targets_dict.to(device)
        src_mask = None
        _, tgt_mask = self.make_mask(out_enc, padded_targets[:, :-1])
        return self.decode(padded_targets[:, :-1], out_enc, src_mask, tgt_mask)

    def forward_test(self, feat, out_enc, img_metas):
        # flatten 2D feature map
        if len(feat.shape) > 3:
            b, c, h, w = feat.shape
            feat = feat.view(b, c, h * w)
            feat = feat.permute((0, 2, 1))
        out_enc = self.feat_positional_encoding(feat) \
            if out_enc is None else out_enc

        batch_size = out_enc.shape[0]
        SOS = torch.zeros(batch_size).long().to(out_enc.device)
        SOS[:] = self.SOS
        SOS = SOS.unsqueeze(1)
        output = self.greedy_forward(SOS, out_enc)
        return output

    def forward(self,
                feat,
                out_enc=None,
                targets_dict=None,
                img_metas=None,
                train_mode=True):
        self.train_mode = train_mode
        if train_mode:
            return self.forward_train(feat, out_enc, targets_dict, img_metas)

        return self.forward_test(feat, out_enc, img_metas)

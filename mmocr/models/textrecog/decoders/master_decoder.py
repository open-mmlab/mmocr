# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import BaseTransformerLayer
from mmcv.runner import ModuleList

from mmocr.models.common.modules import PositionalEncoding
from mmocr.registry import MODELS
from .base_decoder import BaseDecoder


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Embeddings(nn.Module):

    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, *input):
        x = input[0]
        return self.lut(x) * math.sqrt(self.d_model)


@MODELS.register_module()
class MasterDecoder(BaseDecoder):
    """Decoder module in `MASTER <https://arxiv.org/abs/1910.02562>`_.

    Code is partially modified from https://github.com/wenwenyu/MASTER-pytorch.

    Args:
        start_idx (int): The index of `<SOS>`.
        padding_idx (int): The index of `<PAD>`.
        num_classes (int): Number of text characters :math:`C`.
        n_layers (int): Number of attention layers.
        n_head (int): Number of parallel attention heads.
        d_model (int): Dimension :math:`E` of the input from previous model.
        feat_size (int): The size of the input feature from previous model,
            usually :math:`H * W`.
        d_inner (int): Hidden dimension of feedforward layers.
        attn_drop (float): Dropout rate of the attention layer.
        ffn_drop (float): Dropout rate of the feedforward layer.
        feat_pe_drop (float): Dropout rate of the feature positional encoding
            layer.
        max_seq_len (int): Maximum output sequence length :math:`T`.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(
        self,
        start_idx,
        padding_idx,
        num_classes=93,
        n_layers=3,
        n_head=8,
        d_model=512,
        feat_size=6 * 40,
        d_inner=2048,
        attn_drop=0.,
        ffn_drop=0.,
        feat_pe_drop=0.2,
        max_seq_len=30,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        operation_order = ('norm', 'self_attn', 'norm', 'cross_attn', 'norm',
                           'ffn')
        decoder_layer = BaseTransformerLayer(
            operation_order=operation_order,
            attn_cfgs=dict(
                type='MultiheadAttention',
                embed_dims=d_model,
                num_heads=n_head,
                attn_drop=attn_drop,
                dropout_layer=dict(type='Dropout', drop_prob=attn_drop),
            ),
            ffn_cfgs=dict(
                type='FFN',
                embed_dims=d_model,
                feedforward_channels=d_inner,
                ffn_drop=ffn_drop,
                dropout_layer=dict(type='Dropout', drop_prob=ffn_drop),
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
            d_hid=d_model, n_position=self.max_seq_len + 1)
        self.feat_positional_encoding = PositionalEncoding(
            d_hid=d_model, n_position=self.feat_size, dropout=feat_pe_drop)
        self.norm = nn.LayerNorm(d_model)

    def make_mask(self, tgt, device):
        """Make mask for self attention.

        Args:
            tgt (Tensor): Shape [N, l_tgt]
            device (torch.Device): Mask device.

        Returns:
            Tensor: Mask of shape [N * self.n_head, l_tgt, l_tgt]
        """

        trg_pad_mask = (tgt != self.PAD).unsqueeze(1).unsqueeze(3).bool()
        tgt_len = tgt.size(1)
        trg_sub_mask = torch.tril(
            torch.ones((tgt_len, tgt_len), dtype=torch.bool, device=device))
        tgt_mask = trg_pad_mask & trg_sub_mask

        # inverse for mmcv's BaseTransformerLayer
        tril_mask = tgt_mask.clone()
        tgt_mask = tgt_mask.float().masked_fill_(tril_mask == 0, -1e9)
        tgt_mask = tgt_mask.masked_fill_(tril_mask, 0)
        tgt_mask = tgt_mask.repeat(1, self.n_head, 1, 1)
        tgt_mask = tgt_mask.view(-1, tgt_len, tgt_len)
        return tgt_mask

    def decode(self, input, feature, src_mask, tgt_mask):
        x = self.embedding(input)
        x = self.positional_encoding(x)
        attn_masks = [tgt_mask, src_mask]
        for layer in self.decoder_layers:
            x = layer(
                query=x, key=feature, value=feature, attn_masks=attn_masks)
        x = self.norm(x)
        return self.cls(x)

    def greedy_forward(self, SOS, feature):
        input = SOS
        output = None
        for _ in range(self.max_seq_len):
            target_mask = self.make_mask(input, device=feature.device)
            out = self.decode(input, feature, None, target_mask)
            output = out
            prob = F.softmax(out, dim=-1)
            _, next_word = torch.max(prob, dim=-1)
            input = torch.cat([input, next_word[:, -1].unsqueeze(-1)], dim=1)
        return output

    def forward_train(self, feat, out_enc, targets_dict, img_metas=None):
        """
        Args:
            feat (Tensor): The feature map from backbone of shape
                :math:`(N, E, H, W)`.
            out_enc (Tensor): Encoder output.
            targets_dict (dict): A dict with the key ``padded_targets``, a
                tensor of shape :math:`(N, T)`. Each element is the index of a
                character.
            img_metas: Unused.

        Returns:
            Tensor: Raw logit tensor of shape :math:`(N, T, C)`.
        """

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
        tgt_mask = self.make_mask(padded_targets, device=out_enc.device)
        return self.decode(padded_targets, out_enc, src_mask, tgt_mask)

    def forward_test(self, feat, out_enc, img_metas):
        """
        Args:
            feat (Tensor): The feature map from backbone of shape
                :math:`(N, E, H, W)`.
            out_enc (Tensor): Encoder output.
            img_metas: Unused.

        Returns:
            Tensor: Raw logit tensor of shape :math:`(N, T, C)`.
        """

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

# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn

from mmocr.models.builder import DECODERS
from mmocr.models.textrecog.layers import (DotProductAttentionLayer,
                                           PositionAwareLayer)
from .base_decoder import BaseDecoder


@DECODERS.register_module()
class PositionAttentionDecoder(BaseDecoder):
    """Position attention decoder for RobustScanner.

    RobustScanner: `RobustScanner: Dynamically Enhancing Positional Clues for
    Robust Text Recognition <https://arxiv.org/abs/2007.07542>`_

    Args:
        num_classes (int): Number of output classes :math:`C`.
        rnn_layers (int): Number of RNN layers.
        dim_input (int): Dimension :math:`D_i` of input vector ``feat``.
        dim_model (int): Dimension :math:`D_m` of the model. Should also be the
            same as encoder output vector ``out_enc``.
        max_seq_len (int): Maximum output sequence length :math:`T`.
        mask (bool): Whether to mask input features according to
            ``img_meta['valid_ratio']``.
        return_feature (bool): Return feature or logits as the result.
        encode_value (bool): Whether to use the output of encoder ``out_enc``
            as `value` of attention layer. If False, the original feature
            ``feat`` will be used.
        init_cfg (dict or list[dict], optional): Initialization configs.

    Warning:
        This decoder will not predict the final class which is assumed to be
        `<PAD>`. Therefore, its output size is always :math:`C - 1`. `<PAD>`
        is also ignored by loss as specified in
        :obj:`mmocr.models.textrecog.recognizer.EncodeDecodeRecognizer`.
    """

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
        """
        Args:
            feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`.
            out_enc (Tensor): Encoder output of shape
                :math:`(N, D_m, H, W)`.
            targets_dict (dict): A dict with the key ``padded_targets``, a
                tensor of shape :math:`(N, T)`. Each element is the index of a
                character.
            img_metas (dict): A dict that contains meta information of input
                images. Preferably with the key ``valid_ratio``.

        Returns:
            Tensor: A raw logit tensor of shape :math:`(N, T, C-1)` if
            ``return_feature=False``. Otherwise it will be the hidden feature
            before the prediction projection layer, whose shape is
            :math:`(N, T, D_m)`.
        """
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
        """
        Args:
            feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`.
            out_enc (Tensor): Encoder output of shape
                :math:`(N, D_m, H, W)`.
            img_metas (dict): A dict that contains meta information of input
                images. Preferably with the key ``valid_ratio``.

        Returns:
            Tensor: A raw logit tensor of shape :math:`(N, T, C-1)` if
            ``return_feature=False``. Otherwise it would be the hidden feature
            before the prediction projection layer, whose shape is
            :math:`(N, T, D_m)`.
        """
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

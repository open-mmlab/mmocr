# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn

from mmocr.structures import TextRecogDataSample
from mmocr.models.textrecog.dictionary import Dictionary
from mmocr.models.textrecog.layers import (DotProductAttentionLayer,
                                           PositionAwareLayer)
from mmocr.registry import MODELS
from .base_decoder import BaseDecoder


@MODELS.register_module()
class PositionAttentionDecoder(BaseDecoder):
    """Position attention decoder for RobustScanner.

    RobustScanner: `RobustScanner: Dynamically Enhancing Positional Clues for
    Robust Text Recognition <https://arxiv.org/abs/2007.07542>`_

    Args:
        dictionary (dict or :obj:`Dictionary`): The config for `Dictionary` or
            the instance of `Dictionary`.
        module_loss (dict, optional): Config to build module_loss. Defaults
            to None.
        postprocessor (dict, optional): Config to build postprocessor.
            Defaults to None.
        rnn_layers (int): Number of RNN layers. Defaults to 2.
        dim_input (int): Dimension :math:`D_i` of input vector ``feat``.
            Defaults to 512.
        dim_model (int): Dimension :math:`D_m` of the model. Should also be the
            same as encoder output vector ``out_enc``. Defaults to 128.
        max_seq_len (int): Maximum output sequence length :math:`T`. Defaults
            to 40.
        mask (bool): Whether to mask input features according to
            ``img_meta['valid_ratio']``. Defaults to True.
        return_feature (bool): Return feature or logits as the result. Defaults
            to True.
        encode_value (bool): Whether to use the output of encoder ``out_enc``
            as `value` of attention layer. If False, the original feature
            ``feat`` will be used. Defaults to False.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """

    def __init__(
            self,
            dictionary: Union[Dictionary, Dict],
            module_loss: Optional[Dict] = None,
            postprocessor: Optional[Dict] = None,
            rnn_layers: int = 2,
            dim_input: int = 512,
            dim_model: int = 128,
            max_seq_len: int = 40,
            mask: bool = True,
            return_feature: bool = True,
            encode_value: bool = False,
            init_cfg: Optional[Union[Dict, Sequence[Dict]]] = None) -> None:
        super().__init__(
            dictionary=dictionary,
            module_loss=module_loss,
            postprocessor=postprocessor,
            max_seq_len=max_seq_len,
            init_cfg=init_cfg)

        self.dim_input = dim_input
        self.dim_model = dim_model
        self.return_feature = return_feature
        self.encode_value = encode_value
        self.mask = mask

        self.embedding = nn.Embedding(self.max_seq_len + 1, self.dim_model)

        self.position_aware_module = PositionAwareLayer(
            self.dim_model, rnn_layers)

        self.attention_layer = DotProductAttentionLayer()

        self.prediction = None
        if not self.return_feature:
            self.prediction = nn.Linear(
                dim_model if encode_value else dim_input,
                self.dictionary.num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def _get_position_index(
            self,
            length: int,
            batch_size: int,
            device: Optional[torch.device] = None) -> torch.Tensor:
        """Get position index for position attention.

        Args:
            length (int): Length of the sequence.
            batch_size (int): Batch size.
            device (torch.device, optional): Device. Defaults to None.

        Returns:
            torch.Tensor: Position index.
        """
        position_index = torch.arange(0, length, device=device)
        position_index = position_index.repeat([batch_size, 1])
        position_index = position_index.long()
        return position_index

    def forward_train(
            self, feat: torch.Tensor, out_enc: torch.Tensor,
            data_samples: Sequence[TextRecogDataSample]) -> torch.Tensor:
        """
        Args:
            feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`.
            out_enc (Tensor): Encoder output of shape
                :math:`(N, D_m, H, W)`.
            data_samples (list[TextRecogDataSample], optional): Batch of
                TextRecogDataSample, containing gt_text information. Defaults
                to None.

        Returns:
            Tensor: A raw logit tensor of shape :math:`(N, T, C)` if
            ``return_feature=False``. Otherwise it will be the hidden feature
            before the prediction projection layer, whose shape is
            :math:`(N, T, D_m)`.
        """
        valid_ratios = [
            data_sample.get('valid_ratio', 1.0) for data_sample in data_samples
        ] if self.mask else None

        #
        n, c_enc, h, w = out_enc.size()
        assert c_enc == self.dim_model
        _, c_feat, _, _ = feat.size()
        assert c_feat == self.dim_input
        position_index = self._get_position_index(self.max_seq_len, n,
                                                  feat.device)

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
        attn_out = attn_out.permute(0, 2,
                                    1).contiguous()  # [n, max_seq_len, dim_v]

        if self.return_feature:
            return attn_out

        return self.prediction(attn_out)

    def forward_test(self, feat: torch.Tensor, out_enc: torch.Tensor,
                     img_metas: Sequence[TextRecogDataSample]) -> torch.Tensor:
        """
        Args:
            feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`.
            out_enc (Tensor): Encoder output of shape
                :math:`(N, D_m, H, W)`.
            data_samples (list[TextRecogDataSample], optional): Batch of
                TextRecogDataSample, containing gt_text information. Defaults
                to None.

        Returns:
            Tensor: Character probabilities of shape :math:`(N, T, C)` if
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

        return self.softmax(self.prediction(attn_out))

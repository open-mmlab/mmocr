# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmocr.data import TextRecogDataSample
from mmocr.models.textrecog.dictionary import Dictionary
from mmocr.models.textrecog.layers import DotProductAttentionLayer
from mmocr.registry import MODELS
from .base_decoder import BaseDecoder


@MODELS.register_module()
class SequenceAttentionDecoder(BaseDecoder):
    """Sequence attention decoder for RobustScanner.

    RobustScanner: `RobustScanner: Dynamically Enhancing Positional Clues for
    Robust Text Recognition <https://arxiv.org/abs/2007.07542>`_

    Args:
        dictionary (dict or :obj:`Dictionary`): The config for `Dictionary` or
            the instance of `Dictionary`.
        loss_module (dict, optional): Config to build loss_module. Defaults
            to None.
        postprocessor (dict, optional): Config to build postprocessor.
            Defaults to None.
        rnn_layers (int): Number of RNN layers. Defaults to 2.
        dim_input (int): Dimension :math:`D_i` of input vector ``feat``.
            Defaults to 512.
        dim_model (int): Dimension :math:`D_m` of the model. Should also be the
            same as encoder output vector ``out_enc``. Defaults to 128.
        max_seq_len (int): Maximum output sequence length :math:`T`.
            Defaults to 40.
        mask (bool): Whether to mask input features according to
            ``data_sample.valid_ratio``. Defaults to True.
        dropout (float): Dropout rate for LSTM layer. Defaults to 0.
        return_feature (bool): Return feature or logic as the result.
            Defaults to True.
        encode_value (bool): Whether to use the output of encoder ``out_enc``
            as `value` of attention layer. If False, the original feature
            ``feat`` will be used. Defaults to False.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """

    def __init__(self,
                 dictionary: Union[Dictionary, Dict],
                 loss_module: Optional[Dict] = None,
                 postprocessor: Optional[Dict] = None,
                 rnn_layers: int = 2,
                 dim_input: int = 512,
                 dim_model: int = 128,
                 max_seq_len: int = 40,
                 mask: bool = True,
                 dropout: int = 0,
                 return_feature: bool = True,
                 encode_value: bool = False,
                 init_cfg: Optional[Union[Dict,
                                          Sequence[Dict]]] = None) -> None:
        super().__init__(
            dictionary=dictionary,
            loss_module=loss_module,
            postprocessor=postprocessor,
            max_seq_len=max_seq_len,
            init_cfg=init_cfg)

        self.dim_input = dim_input
        self.dim_model = dim_model
        self.return_feature = return_feature
        self.encode_value = encode_value
        self.mask = mask

        self.embedding = nn.Embedding(
            self.dictionary.num_classes,
            self.dim_model,
            padding_idx=self.dictionary.padding_idx)

        self.sequence_layer = nn.LSTM(
            input_size=dim_model,
            hidden_size=dim_model,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout)

        self.attention_layer = DotProductAttentionLayer()

        self.prediction = None
        if not self.return_feature:
            self.prediction = nn.Linear(
                dim_model if encode_value else dim_input,
                self.dictionary.num_classes)

    def forward_train(
        self,
        feat: torch.Tensor,
        out_enc: torch.Tensor,
        data_samples: Optional[Sequence[TextRecogDataSample]] = None
    ) -> torch.Tensor:
        """
        Args:
            feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`.
            out_enc (Tensor): Encoder output of shape
                :math:`(N, D_m, H, W)`.
            targets_dict (dict): A dict with the key ``padded_targets``, a
                tensor of shape :math:`(N, T)`. Each element is the index of a
                character.
            data_samples (list[TextRecogDataSample], optional): Batch of
                TextRecogDataSample, containing gt_text information. Defaults
                to None.

        Returns:
            Tensor: A raw logit tensor of shape :math:`(N, T, C)` if
            ``return_feature=False``. Otherwise it would be the hidden feature
            before the prediction projection layer, whose shape is
            :math:`(N, T, D_m)`.
        """

        valid_ratios = [
            data_sample.get('valid_ratio', 1.0) for data_sample in data_samples
        ] if self.mask else None

        padded_targets = [
            data_sample.gt_text.padded_indexes for data_sample in data_samples
        ]
        padded_targets = torch.stack(padded_targets, dim=0).to(feat.device)
        tgt_embedding = self.embedding(padded_targets)

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

    def forward_test(self, feat: torch.Tensor, out_enc: torch.Tensor,
                     data_samples: Optional[Sequence[TextRecogDataSample]]
                     ) -> torch.Tensor:
        """
        Args:
            feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`.
            out_enc (Tensor): Encoder output of shape
                :math:`(N, D_m, H, W)`.
            data_samples (list[TextRecogDataSample], optional): Batch of
                TextRecogDataSample, containing gt_text information. Defaults
                to None.

        Returns:
            Tensor: The output logit sequence tensor of shape
            :math:`(N, T, C)`.
        """
        seq_len = self.max_seq_len
        batch_size = feat.size(0)

        decode_sequence = (feat.new_ones(
            (batch_size, seq_len)) * self.dictionary.start_idx).long()
        assert not self.return_feature
        outputs = []
        for i in range(seq_len):
            step_out = self.forward_test_step(feat, out_enc, decode_sequence,
                                              i, data_samples)
            outputs.append(step_out)
            _, max_idx = torch.max(step_out, dim=1, keepdim=False)
            if i < seq_len - 1:
                decode_sequence[:, i + 1] = max_idx

        outputs = torch.stack(outputs, 1)

        return outputs

    def forward_test_step(self, feat: torch.Tensor, out_enc: torch.Tensor,
                          decode_sequence: torch.Tensor, current_step: int,
                          data_samples: Sequence[TextRecogDataSample]
                          ) -> torch.Tensor:
        """
        Args:
            feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`.
            out_enc (Tensor): Encoder output of shape
                :math:`(N, D_m, H, W)`.
            decode_sequence (Tensor): Shape :math:`(N, T)`. The tensor that
                stores history decoding result.
            current_step (int): Current decoding step.
            data_samples (list[TextRecogDataSample], optional): Batch of
                TextRecogDataSample, containing gt_text information. Defaults
                to None.

        Returns:
            Tensor: Shape :math:`(N, C)`. The logit tensor of predicted
            tokens at current time step.
        """
        valid_ratios = [
            img_meta.get('valid_ratio', 1.0) for img_meta in data_samples
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

        if not self.return_feature:
            out = self.prediction(out)
            out = F.softmax(out, dim=-1)

        return out

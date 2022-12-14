# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn

from mmocr.models.common.dictionary import Dictionary
from mmocr.registry import MODELS
from mmocr.structures import TextRecogDataSample
from .base import BaseDecoder


@MODELS.register_module()
class ASTERDecoder(BaseDecoder):
    """Implement attention decoder.

    Args:
        in_channels (int): Number of input channels.
        emb_dims (int): Dims of char embedding. Defaults to 512.
        attn_dims (int): Dims of attention. Both hidden states and features
            will be projected to this dims. Defaults to 512.
        hidden_size(int): Dims of hidden state for GRU. Defaults to 512.
        dictionary (dict or :obj:`Dictionary`): The config for `Dictionary` or
            the instance of `Dictionary`. Defaults to None.
        max_seq_len (int): Maximum output sequence length :math:`T`. Defaults
            to 25.
        module_loss (dict, optional): Config to build loss. Defaults to None.
        postprocessor (dict, optional): Config to build postprocessor.
            Defaults to None.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 emb_dims: int = 512,
                 attn_dims: int = 512,
                 hidden_size: int = 512,
                 dictionary: Union[Dictionary, Dict] = None,
                 max_seq_len: int = 25,
                 module_loss: Dict = None,
                 postprocessor: Dict = None,
                 init_cfg=dict(type='Xavier', layer='Conv2d')):
        super().__init__(
            init_cfg=init_cfg,
            dictionary=dictionary,
            module_loss=module_loss,
            postprocessor=postprocessor,
            max_seq_len=max_seq_len)

        self.start_idx = self.dictionary.start_idx
        self.num_classes = self.dictionary.num_classes
        self.in_channels = in_channels
        self.embedding_dim = emb_dims
        self.att_dims = attn_dims
        self.hidden_size = hidden_size

        # Projection layers
        self.proj_feat = nn.Linear(in_channels, attn_dims)
        self.proj_hidden = nn.Linear(hidden_size, attn_dims)
        self.proj_sum = nn.Linear(attn_dims, 1)

        # Decoder input embedding
        self.embedding = nn.Embedding(self.num_classes, self.att_dims)

        # GRU
        self.gru = nn.GRU(
            input_size=self.in_channels + self.embedding_dim,
            hidden_size=self.hidden_size,
            batch_first=True)

        # Prediction layer
        self.fc = nn.Linear(hidden_size, self.dictionary.num_classes)

    def _attention(self, feat: torch.Tensor, prev_hidden: torch.Tensor,
                   prev_char: torch.Tensor):
        """Implement the attention mechanism.

        Args:
            feat (Tensor): Feature map from encoder of shape :math:`(N, T, C)`.
            prev_hidden (Tensor): Previous hidden state from GRU of shape
                :math:`(1, N, self.hidden_size)`.
            prev_char (Tensor): Previous predicted character of shape
                :math:`(N, )`.

        Returns:
            tuple(Tensor, Tensor):
                - output (Tensor): Predicted character of current time step of
                    shape :math:`(N, 1)`.
                - stage (Tensor): Hidden state form GPU of current time step of
                    shape :math:`(N, self.hidden_size)`.
        """
        # Calculate the attention weights
        B, T, _ = feat.size()
        feat_proj = self.proj_feat(feat)  # [N, T, attn_dims]
        hidden_proj = self.proj_hidden(prev_hidden)  # [1, N, attn_dims]
        hidden_proj = hidden_proj.squeeze(0).unsqueeze(1)  # [N, 1, attn_dims]
        hidden_proj = hidden_proj.expand(B, T,
                                         self.att_dims)  # [N, T, attn_dims]

        sum_tanh = torch.tanh(feat_proj + hidden_proj)  # [N, T, attn_dims]
        sum_proj = self.proj_sum(sum_tanh).squeeze()  # [N, T]
        attn_weights = torch.softmax(sum_proj, dim=1)  # [N, T]

        # GRU forward
        context = torch.bmm(attn_weights.unsqueeze(1), feat).squeeze(1)
        char_embed = self.embedding(prev_char.long())  # [N, emb_dims]
        output, state = self.gru(
            torch.cat([char_embed, context], 1).unsqueeze(1), prev_hidden)
        output = output.squeeze(1)
        output = self.fc(output)
        return output, state

    def forward_train(
            self,
            feat: torch.Tensor = None,
            out_enc: Optional[torch.Tensor] = None,
            data_samples: Optional[Sequence[TextRecogDataSample]] = None):
        """
        Args:
            feat (Tensor): Feature from backbone. Unused in this decoder.
            out_enc (torch.Tensor, optional): Encoder output. Defaults to None.
            data_samples (list[TextRecogDataSample], optional): Batch of
                TextRecogDataSample, containing gt_text information. Defaults
                to None.

        Returns:
            Tensor: The raw logit tensor. Shape :math:`(N, T, C)` where
            :math:`C` is ``num_classes``.
        """
        B = out_enc.shape[0]
        state = torch.zeros(1, B, self.hidden_size).to(out_enc.device)
        padded_targets = [
            data_sample.gt_text.padded_indexes for data_sample in data_samples
        ]
        padded_targets = torch.stack(padded_targets, dim=0).to(out_enc.device)
        outputs = []
        for i in range(self.max_seq_len):
            prev_char = padded_targets[:, i].to(out_enc.device)
            output, state = self._attention(out_enc, state, prev_char)
            outputs.append(output)
        outputs = torch.cat([_.unsqueeze(1) for _ in outputs], 1)
        return outputs

    def forward_test(
            self,
            feat: Optional[torch.Tensor] = None,
            out_enc: Optional[torch.Tensor] = None,
            data_samples: Optional[Sequence[TextRecogDataSample]] = None):
        """
        Args:
            feat (Tensor): Feature from backbone. Unused in this decoder.
            out_enc (torch.Tensor, optional): Encoder output. Defaults to None.
            data_samples (list[TextRecogDataSample], optional): Batch of
                TextRecogDataSample, containing gt_text information. Defaults
                to None. Unused in this decoder.

        Returns:
            Tensor: The raw logit tensor. Shape :math:`(N, T, C)` where
            :math:`C` is ``num_classes``.
        """
        B = out_enc.shape[0]
        predicted = []
        state = torch.zeros(1, B, self.hidden_size).to(out_enc.device)
        outputs = []
        for i in range(self.max_seq_len):
            if i == 0:
                prev_char = torch.zeros(B).fill_(self.start_idx).to(
                    out_enc.device)
            else:
                prev_char = predicted

            output, state = self._attention(out_enc, state, prev_char)
            outputs.append(output)
            output = torch.softmax(output, dim=1)
            _, predicted = output.max(1)
        outputs = torch.cat([_.unsqueeze(1) for _ in outputs], 1)
        return outputs

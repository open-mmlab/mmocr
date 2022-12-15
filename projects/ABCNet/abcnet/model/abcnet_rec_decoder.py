# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from mmocr.models.common.dictionary import Dictionary
from mmocr.models.textrecog.decoders.base import BaseDecoder
from mmocr.registry import MODELS
from mmocr.structures import TextRecogDataSample


@MODELS.register_module()
class ABCNetRecDecoder(BaseDecoder):
    """Decoder for ABCNet.

    Args:
        in_channels (int): Number of input channels.
        dropout_prob (float): Probability of dropout. Default to 0.5.
        teach_prob (float): Probability of teacher forcing. Defaults to 0.5.
        dictionary (dict or :obj:`Dictionary`): The config for `Dictionary` or
            the instance of `Dictionary`.
        module_loss (dict, optional): Config to build module_loss. Defaults
            to None.
        postprocessor (dict, optional): Config to build postprocessor.
            Defaults to None.
        max_seq_len (int, optional): Max sequence length. Defaults to 30.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int = 256,
                 dropout_prob: float = 0.5,
                 teach_prob: float = 0.5,
                 dictionary: Union[Dictionary, Dict] = None,
                 module_loss: Dict = None,
                 postprocessor: Dict = None,
                 max_seq_len: int = 30,
                 init_cfg=dict(type='Xavier', layer='Conv2d'),
                 **kwargs):
        super().__init__(
            init_cfg=init_cfg,
            dictionary=dictionary,
            module_loss=module_loss,
            postprocessor=postprocessor,
            max_seq_len=max_seq_len)
        self.in_channels = in_channels
        self.teach_prob = teach_prob
        self.embedding = nn.Embedding(self.dictionary.num_classes, in_channels)
        self.attn_combine = nn.Linear(in_channels * 2, in_channels)
        self.dropout = nn.Dropout(dropout_prob)
        self.gru = nn.GRU(in_channels, in_channels)
        self.out = nn.Linear(in_channels, self.dictionary.num_classes)
        self.vat = nn.Linear(in_channels, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward_train(
        self,
        feat: torch.Tensor,
        out_enc: Optional[torch.Tensor] = None,
        data_samples: Optional[Sequence[TextRecogDataSample]] = None
    ) -> torch.Tensor:
        """
        Args:
            feat (Tensor): A Tensor of shape :math:`(N, C, 1, W)`.
            out_enc (torch.Tensor, optional): Encoder output. Defaults to None.
            data_samples (list[TextRecogDataSample], optional): Batch of
                TextRecogDataSample, containing gt_text information. Defaults
                to None.

        Returns:
            Tensor: The raw logit tensor. Shape :math:`(N, W, C)` where
            :math:`C` is ``num_classes``.
        """
        bs = out_enc.size()[1]
        trg_seq = []
        for target in data_samples:
            trg_seq.append(target.gt_text.padded_indexes.to(feat.device))
        decoder_input = torch.zeros(bs).long().to(out_enc.device)
        trg_seq = torch.stack(trg_seq, dim=0)
        decoder_hidden = torch.zeros(1, bs,
                                     self.in_channels).to(out_enc.device)
        decoder_outputs = []
        for index in range(trg_seq.shape[1]):
            #  decoder_output (nbatch, ncls)
            decoder_output, decoder_hidden = self._attention(
                decoder_input, decoder_hidden, out_enc)
            teach_forcing = True if random.random(
            ) > self.teach_prob else False
            if teach_forcing:
                decoder_input = trg_seq[:, index]  # Teacher forcing
            else:
                _, topi = decoder_output.data.topk(1)
                decoder_input = topi.squeeze()
            decoder_outputs.append(decoder_output)

        return torch.stack(decoder_outputs, dim=1)

    def forward_test(
        self,
        feat: Optional[torch.Tensor] = None,
        out_enc: Optional[torch.Tensor] = None,
        data_samples: Optional[Sequence[TextRecogDataSample]] = None
    ) -> torch.Tensor:
        """
        Args:
            feat (Tensor): A Tensor of shape :math:`(N, C, 1, W)`.
            out_enc (torch.Tensor, optional): Encoder output. Defaults to None.
            data_samples (list[TextRecogDataSample]): Batch of
                TextRecogDataSample, containing ``gt_text`` information.
                Defaults to None.

        Returns:
            Tensor: Character probabilities. of shape
            :math:`(N, self.max_seq_len, C)` where :math:`C` is
            ``num_classes``.
        """
        bs = out_enc.size()[1]
        outputs = []
        decoder_input = torch.zeros(bs).long().to(out_enc.device)
        decoder_hidden = torch.zeros(1, bs,
                                     self.in_channels).to(out_enc.device)
        for _ in range(self.max_seq_len):
            #  decoder_output (nbatch, ncls)
            decoder_output, decoder_hidden = self._attention(
                decoder_input, decoder_hidden, out_enc)
            _, topi = decoder_output.data.topk(1)
            decoder_input = topi.squeeze()
            outputs.append(decoder_output)
        outputs = torch.stack(outputs, dim=1)
        return self.softmax(outputs)

    def _attention(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        # test
        batch_size = encoder_outputs.shape[1]

        alpha = hidden + encoder_outputs
        alpha = alpha.view(-1, alpha.shape[-1])  # (T * n, hidden_size)
        attn_weights = self.vat(torch.tanh(alpha))  # (T * n, 1)
        attn_weights = attn_weights.view(-1, 1, batch_size).permute(
            (2, 1, 0))  # (T, 1, n)  -> (n, 1, T)
        attn_weights = F.softmax(attn_weights, dim=2)

        attn_applied = torch.matmul(attn_weights,
                                    encoder_outputs.permute((1, 0, 2)))

        if embedded.dim() == 1:
            embedded = embedded.unsqueeze(0)
        output = torch.cat((embedded, attn_applied.squeeze(1)), 1)
        output = self.attn_combine(output).unsqueeze(0)  # (1, n, hidden_size)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)  # (1, n, hidden_size)
        output = self.out(output[0])
        return output, hidden

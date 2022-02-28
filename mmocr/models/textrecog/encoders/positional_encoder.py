# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
from torch import nn

from mmocr.models.builder import ENCODERS
from .base_encoder import BaseEncoder


@ENCODERS.register_module()
class PositionalEncoder(BaseEncoder):
    """Implementation of position encoder module in `MASTER.

    <https://arxiv.org/abs/1910.02562>`_.

    Args:
        d_model (int): Dim :math:`D_i` of channels from backbone.
        dropout (float): Dropout probability in encoder.
        max_len (int): max length of feature sequence,
                    should be greater or equal to flatten feature map's length.
    """

    def __init__(self, d_model, dropout=0., max_len=5000):
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -math.log(10000.0) / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, feat, **kwargs):
        if len(feat.shape) > 3:
            b, c, h, w = feat.shape
            feat = feat.view(b, c, h*w)  # flatten 2D feature map
            feat = feat.permute((0, 2, 1))
        feat = feat + self.pe[:, :feat.size(1)]  # pe 1*5000*512
        return self.dropout(feat)
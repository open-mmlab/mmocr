# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmocr.data import TextRecogDataSample
from mmocr.registry import MODELS
from .base_encoder import BaseEncoder


@MODELS.register_module()
class SAREncoder(BaseEncoder):
    """Implementation of encoder module in `SAR.

    <https://arxiv.org/abs/1811.00751>`_.

    Args:
        enc_bi_rnn (bool): If True, use bidirectional RNN in encoder.
            Defaults to False.
        rnn_dropout (float): Dropout probability of RNN layer in encoder.
            Defaults to 0.0.
        enc_gru (bool): If True, use GRU, else LSTM in encoder. Defaults
            to False.
        d_model (int): Dim :math:`D_i` of channels from backbone. Defaults
            to 512.
        d_enc (int): Dim :math:`D_m` of encoder RNN layer. Defaults to 512.
        mask (bool): If True, mask padding in RNN sequence. Defaults to
            True.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to [dict(type='Xavier', layer='Conv2d'),
            dict(type='Uniform', layer='BatchNorm2d')].
    """

    def __init__(self,
                 enc_bi_rnn: bool = False,
                 rnn_dropout: Union[int, float] = 0.0,
                 enc_gru: bool = False,
                 d_model: int = 512,
                 d_enc: int = 512,
                 mask: bool = True,
                 init_cfg: Sequence[Dict] = [
                     dict(type='Xavier', layer='Conv2d'),
                     dict(type='Uniform', layer='BatchNorm2d')
                 ],
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(enc_bi_rnn, bool)
        assert isinstance(rnn_dropout, (int, float))
        assert 0 <= rnn_dropout < 1.0
        assert isinstance(enc_gru, bool)
        assert isinstance(d_model, int)
        assert isinstance(d_enc, int)
        assert isinstance(mask, bool)

        self.enc_bi_rnn = enc_bi_rnn
        self.rnn_dropout = rnn_dropout
        self.mask = mask

        # LSTM Encoder
        kwargs = dict(
            input_size=d_model,
            hidden_size=d_enc,
            num_layers=2,
            batch_first=True,
            dropout=rnn_dropout,
            bidirectional=enc_bi_rnn)
        if enc_gru:
            self.rnn_encoder = nn.GRU(**kwargs)
        else:
            self.rnn_encoder = nn.LSTM(**kwargs)

        # global feature transformation
        encoder_rnn_out_size = d_enc * (int(enc_bi_rnn) + 1)
        self.linear = nn.Linear(encoder_rnn_out_size, encoder_rnn_out_size)

    def forward(
        self,
        feat: torch.Tensor,
        data_samples: Optional[Sequence[TextRecogDataSample]] = None
    ) -> torch.Tensor:
        """
        Args:
            feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`.
            data_samples (list[TextRecogDataSample], optional): Batch of
                TextRecogDataSample, containing valid_ratio information.
                Defaults to None.

        Returns:
            Tensor: A tensor of shape :math:`(N, D_m)`.
        """
        if data_samples is not None:
            assert len(data_samples) == feat.size(0)

        valid_ratios = None
        if data_samples is not None:
            valid_ratios = [
                data_sample.get('valid_ratio', 1.0)
                for data_sample in data_samples
            ] if self.mask else None

        h_feat = feat.size(2)
        feat_v = F.max_pool2d(
            feat, kernel_size=(h_feat, 1), stride=1, padding=0)
        feat_v = feat_v.squeeze(2)  # bsz * C * W
        feat_v = feat_v.permute(0, 2, 1).contiguous()  # bsz * W * C

        holistic_feat = self.rnn_encoder(feat_v)[0]  # bsz * T * C

        if valid_ratios is not None:
            valid_hf = []
            T = holistic_feat.size(1)
            for i, valid_ratio in enumerate(valid_ratios):
                valid_step = min(T, math.ceil(T * valid_ratio)) - 1
                valid_hf.append(holistic_feat[i, valid_step, :])
            valid_hf = torch.stack(valid_hf, dim=0)
        else:
            valid_hf = holistic_feat[:, -1, :]  # bsz * C

        holistic_feat = self.linear(valid_hf)  # bsz * C

        return holistic_feat

# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, List, Optional, Union

import torch.nn as nn
# from mmengine.model import ModuleList
from mmengine.model import ModuleList
from torch import Tensor

from mmocr.models.textrecog.layers import (Adaptive2DPositionalEncoding,
                                           SATRNEncoderLayer)
from mmocr.registry import MODELS
from mmocr.structures import TextRecogDataSample
from .base import BaseEncoder


@MODELS.register_module()
class SATRNEncoder(BaseEncoder):
    """Implement encoder for SATRN, see `SATRN.

    <https://arxiv.org/abs/1910.04396>`_.

    Args:
        n_layers (int): Number of attention layers. Defaults to 12.
        n_head (int): Number of parallel attention heads. Defaults to 8.
        d_k (int): Dimension of the key vector. Defaults to 64.
        d_v (int): Dimension of the value vector. Defaults to 64.
        d_model (int): Dimension :math:`D_m` of the input from previous model.
            Defaults to 512.
        n_position (int): Length of the positional encoding vector. Must be
            greater than ``max_seq_len``. Defaults to 100.
        d_inner (int): Hidden dimension of feedforward layers. Defaults to 256.
        dropout (float): Dropout rate. Defaults to 0.1.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """

    def __init__(self,
                 n_layers: int = 12,
                 n_head: int = 8,
                 d_k: int = 64,
                 d_v: int = 64,
                 d_model: int = 512,
                 n_position: int = 100,
                 d_inner: int = 256,
                 dropout: float = 0.1,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.d_model = d_model
        self.position_enc = Adaptive2DPositionalEncoding(
            d_hid=d_model,
            n_height=n_position,
            n_width=n_position,
            dropout=dropout)
        self.layer_stack = ModuleList([
            SATRNEncoderLayer(
                d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self,
                feat: Tensor,
                data_samples: List[TextRecogDataSample] = None) -> Tensor:
        """Forward propagation of encoder.

        Args:
            feat (Tensor): Feature tensor of shape :math:`(N, D_m, H, W)`.
            data_samples (list[TextRecogDataSample]): Batch of
                TextRecogDataSample, containing `valid_ratio` information.
                Defaults to None.

        Returns:
            Tensor: A tensor of shape :math:`(N, T, D_m)`.
        """
        valid_ratios = [1.0 for _ in range(feat.size(0))]
        if data_samples is not None:
            valid_ratios = [
                data_sample.get('valid_ratio', 1.0)
                for data_sample in data_samples
            ]
        feat = self.position_enc(feat)
        n, c, h, w = feat.size()
        mask = feat.new_zeros((n, h, w))
        for i, valid_ratio in enumerate(valid_ratios):
            valid_width = min(w, math.ceil(w * valid_ratio))
            mask[i, :, :valid_width] = 1
        mask = mask.view(n, h * w)
        feat = feat.view(n, c, h * w)

        output = feat.permute(0, 2, 1).contiguous()
        for enc_layer in self.layer_stack:
            output = enc_layer(output, h, w, mask)
        output = self.layer_norm(output)

        return output

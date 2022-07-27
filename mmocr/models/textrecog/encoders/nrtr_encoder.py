# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
from mmengine.model import ModuleList

from mmocr.models.common import TFEncoderLayer
from mmocr.registry import MODELS
from mmocr.structures import TextRecogDataSample
from .base_encoder import BaseEncoder


@MODELS.register_module()
class NRTREncoder(BaseEncoder):
    """Transformer Encoder block with self attention mechanism.

    Args:
        n_layers (int): The number of sub-encoder-layers in the encoder.
            Defaults to 6.
        n_head (int): The number of heads in the multiheadattention models
            Defaults to 8.
        d_k (int): Total number of features in key. Defaults to 64.
        d_v (int): Total number of features in value. Defaults to 64.
        d_model (int): The number of expected features in the decoder inputs.
            Defaults to 512.
        d_inner (int): The dimension of the feedforward network model.
            Defaults to 256.
        dropout (float): Dropout rate for MHSA and FFN. Defaults to 0.1.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 n_layers: int = 6,
                 n_head: int = 8,
                 d_k: int = 64,
                 d_v: int = 64,
                 d_model: int = 512,
                 d_inner: int = 256,
                 dropout: float = 0.1,
                 init_cfg: Optional[Union[Dict,
                                          Sequence[Dict]]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.d_model = d_model
        self.layer_stack = ModuleList([
            TFEncoderLayer(
                d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)

    def _get_source_mask(self, src_seq: torch.Tensor,
                         valid_ratios: Sequence[float]) -> torch.Tensor:
        """Generate mask for source sequence.

        Args:
            src_seq (torch.Tensor): Image sequence. Shape :math:`(N, T, C)`.
            valid_ratios (list[float]): The valid ratio of input image. For
                example, if the width of the original image is w1 and the width
                after pad is w2, then valid_ratio = w1/w2. source mask is used
                to cover the area of the pad region.

        Returns:
            Tensor or None: Source mask. Shape :math:`(N, T)`. The region of
            pad area are False, and the rest are True.
        """

        N, T, _ = src_seq.size()
        mask = None
        if len(valid_ratios) > 0:
            mask = src_seq.new_zeros((N, T), device=src_seq.device)
            for i, valid_ratio in enumerate(valid_ratios):
                valid_width = min(T, math.ceil(T * valid_ratio))
                mask[i, :valid_width] = 1

        return mask

    def forward(self,
                feat: torch.Tensor,
                data_samples: Sequence[TextRecogDataSample] = None
                ) -> torch.Tensor:
        """
        Args:
            feat (Tensor): Backbone output of shape :math:`(N, C, H, W)`.
            data_samples (list[TextRecogDataSample]): Batch of
                TextRecogDataSample, containing valid_ratio information.
                Defaults to None.


        Returns:
            Tensor: The encoder output tensor. Shape :math:`(N, T, C)`.
        """
        n, c, h, w = feat.size()

        feat = feat.view(n, c, h * w).permute(0, 2, 1).contiguous()

        valid_ratios = []
        for data_sample in data_samples:
            valid_ratios.append(data_sample.get('valid_ratio'))
        mask = self._get_source_mask(feat, valid_ratios)

        output = feat
        for enc_layer in self.layer_stack:
            output = enc_layer(output, mask)
        output = self.layer_norm(output)

        return output

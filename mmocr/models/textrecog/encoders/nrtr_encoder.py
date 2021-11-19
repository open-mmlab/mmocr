# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch.nn as nn
from mmcv.runner import ModuleList

from mmocr.models.builder import ENCODERS
from mmocr.models.common import TFEncoderLayer
from .base_encoder import BaseEncoder


@ENCODERS.register_module()
class NRTREncoder(BaseEncoder):
    """Encode 2d feature map to 1d sequence."""

    def __init__(self,
                 n_layers=6,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 d_model=512,
                 d_inner=256,
                 dropout=0.1,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.d_model = d_model
        self.layer_stack = ModuleList([
            TFEncoderLayer(
                d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)

    def _get_mask(self, logit, img_metas):
        valid_ratios = None
        if img_metas is not None:
            valid_ratios = [
                img_meta.get('valid_ratio', 1.0) for img_meta in img_metas
            ]
        N, T, _ = logit.size()
        mask = None
        if valid_ratios is not None:
            mask = logit.new_zeros((N, T))
            for i, valid_ratio in enumerate(valid_ratios):
                valid_width = min(T, math.ceil(T * valid_ratio))
                mask[i, :valid_width] = 1

        return mask

    def forward(self, feat, img_metas=None):
        mask = self._get_mask(feat, img_metas)

        output = feat
        for enc_layer in self.layer_stack:
            output = enc_layer(output, mask)
        output = self.layer_norm(output)

        return output

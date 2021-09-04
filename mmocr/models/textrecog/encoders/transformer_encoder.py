# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch.nn as nn
from mmcv.runner import ModuleList

from mmocr.models.builder import ENCODERS
from mmocr.models.textrecog.layers import TransformerEncoderLayer
from .base_encoder import BaseEncoder


@ENCODERS.register_module()
class TFEncoder(BaseEncoder):
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
            TransformerEncoderLayer(
                d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, feat, img_metas=None):
        valid_ratios = [1.0 for _ in range(feat.size(0))]
        if img_metas is not None:
            valid_ratios = [
                img_meta.get('valid_ratio', 1.0) for img_meta in img_metas
            ]
        n, c, h, w = feat.size()
        mask = feat.new_zeros((n, h, w))
        for i, valid_ratio in enumerate(valid_ratios):
            valid_width = min(w, math.ceil(w * valid_ratio))
            mask[i, :, :valid_width] = 1
        mask = mask.view(n, h * w)
        feat = feat.view(n, c, h * w)

        output = feat.permute(0, 2, 1).contiguous()
        for enc_layer in self.layer_stack:
            output = enc_layer(output, mask)
        output = self.layer_norm(output)

        output = output.permute(0, 2, 1).contiguous()
        output = output.view(n, self.d_model, h, w)

        return output

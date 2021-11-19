# Copyright (c) OpenMMLab. All rights reserved.
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

    def forward(self, feat, img_metas=None):

        output = feat
        for enc_layer in self.layer_stack:
            output = enc_layer(output)
        output = self.layer_norm(output)

        return output

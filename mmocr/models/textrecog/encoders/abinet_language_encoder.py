# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from mmocr.models.builder import ENCODERS
from mmocr.models.textrecog.layers import PositionAttention
from .base_encoder import BaseEncoder


@ENCODERS.register_module()
class ABIVisionEncoder(BaseEncoder):

    def __init__(self,
                 in_channels=512,
                 num_channels=64,
                 attn_height=8,
                 attn_width=32,
                 attn_mode='nearest',
                 max_seq_len=40,
                 num_chars=90,
                 init_cfg=dict(type='Xavier', layer='Conv2d')):
        super().__init__(init_cfg=init_cfg)

        self.attention = PositionAttention(
            max_length=max_seq_len + 1,  # additional stop token
            in_channels=in_channels,
            num_channels=num_channels,
            mode=attn_mode,
            h=attn_height,
            w=attn_width,
        )
        self.cls = nn.Linear(in_channels, num_chars)

    def forward(self, feat, img_metas=None):
        attn_vecs, attn_scores = self.attention(feat)
        logits = self.cls(attn_vecs)

        return {
            'feature': attn_vecs,
            'logits': logits,
            'attn_scores': attn_scores
        }

# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from mmocr.models.builder import ENCODERS
from .base_encoder import BaseEncoder


@ENCODERS.register_module()
class ChannelReductionEncoder(BaseEncoder):

    def __init__(self,
                 in_channels,
                 out_channels,
                 init_cfg=dict(type='Xavier', layer='Conv2d')):
        super().__init__(init_cfg=init_cfg)

        self.layer = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, feat, img_metas=None):
        return self.layer(feat)

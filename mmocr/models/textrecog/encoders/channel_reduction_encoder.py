# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from mmocr.models.builder import ENCODERS
from .base_encoder import BaseEncoder


@ENCODERS.register_module()
class ChannelReductionEncoder(BaseEncoder):
    """Change the channel number with a one by one convoluational layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 init_cfg=dict(type='Xavier', layer='Conv2d')):
        super().__init__(init_cfg=init_cfg)

        self.layer = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, feat, img_metas=None):
        """
        Args:
            feat (Tensor): Image features with the shape of
                :math:`(N, C_{in}, H, W)`.
            img_metas (None): Unused.

        Returns:
            Tensor: A tensor of shape :math:`(N, C_{out}, H, W)`.
        """
        return self.layer(feat)

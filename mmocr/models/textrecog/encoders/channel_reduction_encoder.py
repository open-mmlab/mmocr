# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from mmocr.registry import MODELS
from mmocr.structures import TextRecogDataSample
from .base import BaseEncoder


@MODELS.register_module()
class ChannelReductionEncoder(BaseEncoder):
    """Change the channel number with a one by one convoluational layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to dict(type='Xavier', layer='Conv2d').
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        init_cfg: Dict = dict(type='Xavier', layer='Conv2d')
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.layer = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(
        self,
        feat: torch.Tensor,
        data_samples: Optional[Sequence[TextRecogDataSample]] = None
    ) -> torch.Tensor:
        """
        Args:
            feat (Tensor): Image features with the shape of
                :math:`(N, C_{in}, H, W)`.
            data_samples (list[TextRecogDataSample], optional): Batch of
                TextRecogDataSample, containing valid_ratio information.
                Defaults to None.

        Returns:
            Tensor: A tensor of shape :math:`(N, C_{out}, H, W)`.
        """
        return self.layer(feat)

# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence

import torch

from mmocr.models.textrecog.encoders.base import BaseEncoder
from mmocr.models.textrecog.layers import BidirectionalLSTM
from mmocr.registry import MODELS
from mmocr.structures import TextRecogDataSample


@MODELS.register_module()
class ABCNetRecEncoder(BaseEncoder):
    """Encoder for ABCNet.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to dict(type='Xavier', layer='Conv2d').
    """

    def __init__(self,
                 in_channels: int = 256,
                 hidden_channels: int = 256,
                 out_channels: int = 256,
                 init_cfg: Dict = None) -> None:
        super().__init__(init_cfg=init_cfg)

        self.layer = BidirectionalLSTM(in_channels, hidden_channels,
                                       out_channels)

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
        assert feat.size(2) == 1, 'feature height must be 1'
        feat = feat.squeeze(2)
        feat = feat.permute(2, 0, 1)  # NxCxW -> WxNxC
        feat = self.layer(feat)
        # feat = feat.permute(1, 0, 2).contiguous()
        return feat

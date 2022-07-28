# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmocr.registry import MODELS
from .base_encoder import BaseEncoder


@MODELS.register_module()
class ASTEREncoder(BaseEncoder):
    """Implement BiLSTM encoder module in `ASTER: An Attentional Scene Text
    Recognizer with Flexible Rectification.

    <https://ieeexplore.ieee.org/abstract/document/8395027/`

    Args:
        in_channels (int): Number of input channels.
        num_layers (int): Layers of BiLSTM. Defaults to 2.
    """

    def __init__(self,
                 in_channels: int,
                 num_layers: int = 2,
                 init_cfg=dict(type='Xavier', layer='Conv2d')) -> None:
        super().__init__(init_cfg=init_cfg)
        self.bilstm = nn.LSTM(
            in_channels,
            in_channels // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True)

    def forward(self, feat: torch.Tensor, img_metas=None) -> torch.Tensor:
        """
        Args:
            feat (Tensor): Feature of shape (N, C, 1, W).

        Returns:
            Tensor: Output of BiLSTM.
        """
        assert feat.dim() == 4
        assert feat.size(2) == 1, 'height must be 1'
        feat = feat.squeeze(2).permute(0, 2, 1)
        feat, _ = self.bilstm(feat)
        return feat.contiguous()

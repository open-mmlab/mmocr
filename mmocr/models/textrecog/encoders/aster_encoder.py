# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import Sequential
from mmocr.models.builder import ENCODERS
from mmocr.models.textrecog.layers import BidirectionalLSTM
from .base_encoder import BaseEncoder


@ENCODERS.register_module()
class ASTEREncoder(BaseEncoder):

    def __init__(self,
                 in_channels=None,
                 num_classes=None,
                 with_lstm=True,
                 init_cfg=dict(type='Xavier', layer='Conv2d'),
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.with_lstm = with_lstm
        self.encoder = Sequential(
            BidirectionalLSTM(in_channels, 256, 256),
            BidirectionalLSTM(256, 256, num_classes))

    def forward(self, feat, img_metas=None):
        """
        Args:
            feat (Tensor): A Tensor of shape :math:`(N, C, 1, W)`.

        Returns:
            Tensor: The raw logit tensor. Shape :math:`(N, W, C)` where
            :math:`C` is ``num_classes``.

        """
        assert feat.size(2) == 1, 'feature height must be 1'
        if self.with_lstm:
            x = feat.squeeze(2)  # [N, c, w]
            x = x.permute(2, 0, 1)
            x = self.encoder(x)
            outputs = x.permute(1, 0, 2).contiguous()
        else:
            x = feat.permute(0, 3, 1, 2).contiguous()
            n, w, c, h = x.size()
            outputs = x.view(n, w, c * h)
        return outputs

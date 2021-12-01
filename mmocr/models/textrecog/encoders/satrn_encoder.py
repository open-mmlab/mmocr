import math

import torch.nn as nn
from mmcv.runner import ModuleList

from mmocr.models.builder import ENCODERS
from mmocr.models.textrecog.layers import (Adaptive2DPositionalEncoding,
                                           SatrnEncoderLayer)
from .base_encoder import BaseEncoder


@ENCODERS.register_module()
class SatrnEncoder(BaseEncoder):
    """Implement encoder for SATRN, see `SATRN.

    <https://arxiv.org/abs/1910.04396>`_.

    Args:
        n_layers (int): Number of attention layers.
        n_head (int): Number of parallel attention heads.
        d_k (int): Dimension of the key vector.
        d_v (int): Dimension of the value vector.
        d_model (int): Dimension :math:`D_m` of the input from previous model.
        n_position (int): Length of the positional encoding vector. Must be
            greater than ``max_seq_len``.
        d_inner (int): Hidden dimension of feedforward layers.
        dropout (float): Dropout rate.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 n_layers=12,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 d_model=512,
                 n_position=100,
                 d_inner=256,
                 dropout=0.1,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.d_model = d_model
        self.position_enc = Adaptive2DPositionalEncoding(
            d_hid=d_model,
            n_height=n_position,
            n_width=n_position,
            dropout=dropout)
        self.layer_stack = ModuleList([
            SatrnEncoderLayer(
                d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, feat, img_metas=None):
        """
        Args:
            feat (Tensor): Feature tensor of shape :math:`(N, D_m, H, W)`.
            img_metas (dict): A dict that contains meta information of input
                images. Preferably with the key ``valid_ratio``.

        Returns:
            Tensor: A tensor of shape :math:`(N, T, D_m)`.
        """
        valid_ratios = [1.0 for _ in range(feat.size(0))]
        if img_metas is not None:
            valid_ratios = [
                img_meta.get('valid_ratio', 1.0) for img_meta in img_metas
            ]
        feat += self.position_enc(feat)
        n, c, h, w = feat.size()
        mask = feat.new_zeros((n, h, w))
        for i, valid_ratio in enumerate(valid_ratios):
            valid_width = min(w, math.ceil(w * valid_ratio))
            mask[i, :, :valid_width] = 1
        mask = mask.view(n, h * w)
        feat = feat.view(n, c, h * w)

        output = feat.permute(0, 2, 1).contiguous()
        for enc_layer in self.layer_stack:
            output = enc_layer(output, h, w, mask)
        output = self.layer_norm(output)

        return output

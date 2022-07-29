# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Union

import torch
from mmcv.cnn.bricks.transformer import BaseTransformerLayer
from mmengine.model import BaseModule, ModuleList

from mmocr.data import TextRecogDataSample
from mmocr.models.common.modules import PositionalEncoding
from mmocr.registry import MODELS


@MODELS.register_module()
class ABIEncoder(BaseModule):
    """Implement transformer encoder for text recognition, modified from
    `<https://github.com/FangShancheng/ABINet>`.

    Args:
        n_layers (int): Number of attention layers. Defaults to 2.
        n_head (int): Number of parallel attention heads. Defaults to 8.
        d_model (int): Dimension :math:`D_m` of the input from previous model.
            Defaults to 512.
        d_inner (int): Hidden dimension of feedforward layers. Defaults to
            2048.
        dropout (float): Dropout rate. Defaults to 0.1.
        max_len (int): Maximum output sequence length :math:`T`. Defaults to
            8 * 32.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """

    def __init__(self,
                 n_layers: int = 2,
                 n_head: int = 8,
                 d_model: int = 512,
                 d_inner: int = 2048,
                 dropout: float = 0.1,
                 max_len: int = 8 * 32,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None):
        super().__init__(init_cfg=init_cfg)

        assert d_model % n_head == 0, 'd_model must be divisible by n_head'

        self.pos_encoder = PositionalEncoding(d_model, n_position=max_len)
        encoder_layer = BaseTransformerLayer(
            operation_order=('self_attn', 'norm', 'ffn', 'norm'),
            attn_cfgs=dict(
                type='MultiheadAttention',
                embed_dims=d_model,
                num_heads=n_head,
                attn_drop=dropout,
                dropout_layer=dict(type='Dropout', drop_prob=dropout),
            ),
            ffn_cfgs=dict(
                type='FFN',
                embed_dims=d_model,
                feedforward_channels=d_inner,
                ffn_drop=dropout,
            ),
            norm_cfg=dict(type='LN'),
        )
        self.transformer = ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(n_layers)])

    def forward(self, feature: torch.Tensor,
                data_samples: List[TextRecogDataSample]) -> torch.Tensor:
        """
        Args:
            feature (Tensor): Feature tensor of shape :math:`(N, D_m, H, W)`.
            data_samples (List[TextRecogDataSample]): List of data samples.

        Returns:
            Tensor: Features of shape :math:`(N, D_m, H, W)`.
        """
        n, c, h, w = feature.shape
        feature = feature.view(n, c, -1).transpose(1, 2)  # (n, h*w, c)
        feature = self.pos_encoder(feature)  # (n, h*w, c)
        feature = feature.transpose(0, 1)  # (h*w, n, c)
        for m in self.transformer:
            feature = m(feature)
        feature = feature.permute(1, 2, 0).view(n, c, h, w)
        return feature

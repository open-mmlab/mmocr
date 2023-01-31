# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch.nn as nn
from torch import Tensor

from mmocr.models.textrecog.encoders import BaseEncoder
from mmocr.registry import MODELS
from mmocr.structures import TextSpottingDataSample


@MODELS.register_module()
class SPTSEncoder(BaseEncoder):
    """SPTS Encoder.

    Args:
        d_backbone (int): Backbone output dimension.
        d_model (int): Model output dimension.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """

    def __init__(self,
                 d_backbone: int = 2048,
                 d_model: int = 256,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.input_proj = nn.Conv2d(d_backbone, d_model, kernel_size=1)

    def forward(self,
                feat: Tensor,
                data_samples: List[TextSpottingDataSample] = None) -> Tensor:
        """Forward propagation of encoder.

        Args:
            feat (Tensor): Feature tensor of shape :math:`(N, D_m, H, W)`.
            data_samples (list[TextSpottingDataSample]): Batch of
                TextSpottingDataSample.
                Defaults to None.

        Returns:
            Tensor: A tensor of shape :math:`(N, T, D_m)`.
        """
        return self.input_proj(feat[0])

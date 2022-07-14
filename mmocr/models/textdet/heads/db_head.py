# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.runner import Sequential
from torch import Tensor

from mmocr.data import TextDetDataSample
from mmocr.models.textdet.heads import BaseTextDetHead
from mmocr.registry import MODELS


@MODELS.register_module()
class DBHead(BaseTextDetHead):
    """The class for DBNet head.

    This was partially adapted from https://github.com/MhLiao/DB

    Args:
        in_channels (int): The number of input channels.
        with_bias (bool): Whether add bias in Conv2d layer. Defaults to False.
        module_loss (dict): Config of loss for dbnet. Defaults to
            ``dict(type='DBModuleLoss')``
        postprocessor (dict): Config of postprocessor for dbnet.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(
        self,
        in_channels: int,
        with_bias: bool = False,
        module_loss: Dict = dict(type='DBModuleLoss'),
        postprocessor: Dict = dict(
            type='DBPostprocessor', text_repr_type='quad'),
        init_cfg: Optional[Union[Dict, List[Dict]]] = [
            dict(type='Kaiming', layer='Conv'),
            dict(type='Constant', layer='BatchNorm', val=1., bias=1e-4)
        ]
    ) -> None:
        super().__init__(
            module_loss=module_loss,
            postprocessor=postprocessor,
            init_cfg=init_cfg)

        assert isinstance(in_channels, int)
        assert isinstance(with_bias, bool)

        self.in_channels = in_channels
        self.binarize = Sequential(
            nn.Conv2d(
                in_channels, in_channels // 4, 3, bias=with_bias, padding=1),
            nn.BatchNorm2d(in_channels // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2),
            nn.BatchNorm2d(in_channels // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, 2, 2), nn.Sigmoid())
        self.threshold = self._init_thr(in_channels)

    def _diff_binarize(self, prob_map: Tensor, thr_map: Tensor,
                       k: int) -> Tensor:
        """Differential binarization.

        Args:
            prob_map (Tensor): Probability map.
            thr_map (Tensor): Threshold map.
            k (int): Amplification factor.

        Returns:
            Tensor: Binary map.
        """
        return torch.reciprocal(1.0 + torch.exp(-k * (prob_map - thr_map)))

    def forward(
        self,
        img: Tensor,
        data_samples: Optional[List[TextDetDataSample]] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            img (Tensor): Shape :math:`(N, C, H, W)`.
            data_samples (list[TextDetDataSample], optional): A list of data
                samples. Defaults to None.

        Returns:
            tuple(Tensor, Tensor, Tensor): A tuple of ``prob_map``, ``thr_map``
            and ``binary_map``, each of shape :math:`(N, 4H, 4W)`.
        """
        prob_map = self.binarize(img).squeeze(1)
        thr_map = self.threshold(img).squeeze(1)
        binary_map = self._diff_binarize(prob_map, thr_map, k=50).squeeze(1)
        return (prob_map, thr_map, binary_map)

    def _init_thr(self,
                  inner_channels: int,
                  bias: bool = False) -> nn.ModuleList:
        """Initialize threshold branch."""
        in_channels = inner_channels
        seq = Sequential(
            nn.Conv2d(
                in_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.BatchNorm2d(inner_channels // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, inner_channels // 4, 2, 2),
            nn.BatchNorm2d(inner_channels // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, 1, 2, 2), nn.Sigmoid())
        return seq

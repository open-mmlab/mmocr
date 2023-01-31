# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import torch.nn as nn
from mmcv import ops
from torch import Tensor

from mmengine.model import BaseModule
from mmocr.utils import ConfigType, OptMultiConfig


class BaseRoIExtractor(BaseModule, metaclass=ABCMeta):
    """Base class for RoI extractor.

    Args:
        roi_layer (:obj:`ConfigDict` or dict): Specify RoI layer type and
            arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (list[int]): Strides of input feature maps.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 roi_layer: ConfigType,
                 out_channels: int,
                 featmap_strides: List[int],
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides

    @property
    def num_inputs(self) -> int:
        """int: Number of input feature maps."""
        return len(self.featmap_strides)

    def build_roi_layers(self, layer_cfg: ConfigType,
                         featmap_strides: List[int]) -> nn.ModuleList:
        """Build RoI operator to extract feature from each level feature map.

        Args:
            layer_cfg (:obj:`ConfigDict` or dict): Dictionary to construct and
                config RoI layer operation. Options are modules under
                ``mmcv/ops`` such as ``RoIAlign``.
            featmap_strides (list[int]): The stride of input feature map w.r.t
                to the original image size, which would be used to scale RoI
                coordinate (original image coordinate system) to feature
                coordinate system.

        Returns:
            :obj:`nn.ModuleList`: The RoI extractor modules for each level
                feature map.
        """

        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        assert hasattr(ops, layer_type)
        layer_cls = getattr(ops, layer_type)
        roi_layers = nn.ModuleList(
            [layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        return roi_layers

    @abstractmethod
    def forward(self, feats: Tuple[Tensor], data_samples) -> Tensor:
        """Extractor ROI feats.

        Args:
            feats (Tuple[Tensor]): Multi-scale features.
            data_samples (List[TextSpottingDataSample]):

              - proposals(InstanceData): The proposals of text detection.

        Returns:
            Tensor: RoI feature.
        """
        pass

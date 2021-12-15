# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

from mmcv.runner import BaseModule
from mmdet.core import build_assigner, build_sampler


class BaseRoIExtractor(BaseModule, metaclass=ABCMeta):
    """Base class for RoI extractor.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
        train_cfg (mmcv.Config): build for assigner and sampler for training
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 train_cfg=None,
                 init_cfg=None):
        super(BaseRoIExtractor, self).__init__(init_cfg)
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        if train_cfg:
            self.bbox_assigner = build_assigner(train_cfg.assigner)
            self.bbox_sampler = build_sampler(train_cfg.sampler, context=self)

    @property
    def num_inputs(self):
        """int: Number of input feature maps."""
        return len(self.featmap_strides)

    @abstractmethod
    def build_roi_layers(self, layer_cfg, featmap_strides):
        """Build RoI operator to extract feature from each level feature map.

        Returns:
            nn.ModuleList: The RoI extractor modules for each level feature
                map.
        """
        pass

    @abstractmethod
    def forward(self, feats, rois, roi_scale_factor=None):
        pass

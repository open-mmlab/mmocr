# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
from torch import Tensor

from mmengine.structures import InstanceData
from mmocr.registry import MODELS
from mmocr.utils import ConfigType, OptMultiConfig
from .base_roi_extractor import BaseRoIExtractor


@MODELS.register_module()
class BezierRoIExtractor(BaseRoIExtractor):
    """Extract RoI features from a single level feature map.

    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale. The mapping rule is proposed in
    `FPN <https://arxiv.org/abs/1612.03144>`_.

    Args:
        roi_layer (:obj:`ConfigDict` or dict): Specify RoI layer type and
            arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (List[int]): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0.
            Defaults to 56.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 roi_layer: ConfigType,
                 out_channels: int,
                 featmap_strides: List[int],
                 finest_scale: int = 96,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            roi_layer=roi_layer,
            out_channels=out_channels,
            featmap_strides=featmap_strides,
            init_cfg=init_cfg)
        self.finest_scale = finest_scale

    def to_roi(self, beziers: Tensor) -> Tensor:
        rois_list = []
        for img_id, bezier in enumerate(beziers):
            img_inds = bezier.new_full((bezier.size(0), 1), img_id)
            rois = torch.cat([img_inds, bezier], dim=-1)
            rois_list.append(rois)
        rois = torch.cat(rois_list, 0)
        return rois

    def map_roi_levels(self, beziers: Tensor, num_levels: int) -> Tensor:
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3
        Args:
            beziers (Tensor): Input bezier control points, shape (k, 17).
            num_levels (int): Total level number.
        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """

        p1 = beziers[:, 1:3]
        p2 = beziers[:, 15:]
        scale = ((p1 - p2)**2).sum(dim=1).sqrt() * 2
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    def forward(self, feats: Tuple[Tensor],
                proposal_instances: List[InstanceData]) -> Tensor:
        """Extractor ROI feats.

        Args:
            feats (Tuple[Tensor]): Multi-scale features.
            proposal_instances(List[InstanceData]): Proposal instances.

        Returns:
            Tensor: RoI feature.
        """
        beziers = [p_i.beziers for p_i in proposal_instances]
        rois = self.to_roi(beziers)
        # convert fp32 to fp16 when amp is on
        rois = rois.type_as(feats[0])
        out_size = self.roi_layers[0].output_size
        feats = feats[:3]
        num_levels = len(feats)
        roi_feats = feats[0].new_zeros(
            rois.size(0), self.out_channels, *out_size)

        if num_levels == 1:
            if len(rois) == 0:
                return roi_feats
            return self.roi_layers[0](feats[0], rois)

        target_lvls = self.map_roi_levels(rois, num_levels)

        for i in range(num_levels):
            mask = target_lvls == i
            inds = mask.nonzero(as_tuple=False).squeeze(1)
            if inds.numel() > 0:
                rois_ = rois[inds]
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[inds] = roi_feats_t
            else:
                # Sometimes some pyramid levels will not be used for RoI
                # feature extraction and this will cause an incomplete
                # computation graph in one GPU, which is different from those
                # in other GPUs and will cause a hanging error.
                # Therefore, we add it to ensure each feature pyramid is
                # included in the computation graph to avoid runtime bugs.
                roi_feats += sum(
                    x.view(-1)[0]
                    for x in self.parameters()) * 0. + feats[i].sum() * 0.
        return roi_feats

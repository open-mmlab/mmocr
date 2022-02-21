# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import force_fp32
from mmdet.models.roi_heads.roi_extractors import SingleRoIExtractor

from mmocr.models.builder import ROI_EXTRACTORS


@ROI_EXTRACTORS.register_module()
class BezierRoIExtractor(SingleRoIExtractor):
    """Extract RoI features from a single level feature map.

    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale. The mapping rule is proposed in
    `FPN <https://arxiv.org/abs/1612.03144>`_.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (List[int]): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0. Default: 56.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 finest_scale=56,
                 init_cfg=None):
        super(BezierRoIExtractor,
              self).__init__(roi_layer, out_channels, featmap_strides,
                             finest_scale, init_cfg)

    def map_roi_levels(self, beziers, num_levels):
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
        scale = ((p1 - p2)**2).sum(dim=1).sqrt()
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, bezier_rois):
        """Forward function.

        Args:
            feats (tuple(Tensor)): Multi-level features. Each level of the
            feature has the shape of :math:`(N, C, H, W)`.
            bezier_rois (Tensor): The tensor representing RoIs of shape
                :math:`(N, 17)`, where the first element is batch_id and the
                others are Bezier control points representing the RoI.
        """

        return super(BezierRoIExtractor, self).forward(feats, bezier_rois)

# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch import nn

from mmocr.registry import MODELS


@MODELS.register_module()
class SegHead(BaseModule):
    """Head for segmentation based text recognition.

    Args:
        in_channels (int): Number of input channels :math:`C`.
        num_classes (int): Number of output classes :math:`C_{out}`.
        upsample_param (dict | None): Config dict for interpolation layer.
            Default: ``dict(scale_factor=1.0, mode='nearest')``
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 in_channels=128,
                 num_classes=37,
                 upsample_param=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert isinstance(num_classes, int)
        assert num_classes > 0
        assert upsample_param is None or isinstance(upsample_param, dict)

        self.upsample_param = upsample_param

        self.seg_conv = ConvModule(
            in_channels,
            in_channels,
            3,
            stride=1,
            padding=1,
            norm_cfg=dict(type='BN'))

        # prediction
        self.pred_conv = nn.Conv2d(
            in_channels, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, out_neck):
        """
        Args:
            out_neck (list[Tensor]): A list of tensor of shape
                :math:`(N, C_i, H_i, W_i)`. The network only uses the last one
                (``out_neck[-1]``).

        Returns:
            Tensor: A tensor of shape :math:`(N, C_{out}, kH, kW)` where
            :math:`k` is determined by ``upsample_param``.
        """

        seg_map = self.seg_conv(out_neck[-1])
        seg_map = self.pred_conv(seg_map)

        if self.upsample_param is not None:
            seg_map = F.interpolate(seg_map, **self.upsample_param)

        return seg_map

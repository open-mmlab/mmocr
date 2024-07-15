# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmocr.registry import MODELS
from mmocr.utils import ConfigType, MultiConfig, OptConfigType


@MODELS.register_module()
class BiFPN(BaseModule):
    """illustration of a minimal bifpn unit P7_0 ------------------------->
    P7_2 -------->

    |-------------|                ↑                  ↓                |
    P6_0 ---------> P6_1 ---------> P6_2 -------->
    |-------------|--------------↑ ↑                  ↓                | P5_0
    ---------> P5_1 ---------> P5_2 -------->    |-------------|--------------↑
    ↑                  ↓                | P4_0 ---------> P4_1 ---------> P4_2
    -------->    |-------------|--------------↑ ↑
    |--------------↓ | P3_0 -------------------------> P3_2 -------->
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: int,
                 num_outs: int,
                 repeat_times: int = 2,
                 start_level: int = 0,
                 end_level: int = -1,
                 add_extra_convs: bool = False,
                 relu_before_extra_convs: bool = False,
                 no_norm_on_lateral: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 act_cfg: OptConfigType = None,
                 laterial_conv1x1: bool = False,
                 upsample_cfg: ConfigType = dict(mode='nearest'),
                 pool_cfg: ConfigType = dict(),
                 init_cfg: MultiConfig = dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.upsample_cfg = upsample_cfg.copy()
        self.repeat_times = repeat_times
        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs

        self.lateral_convs = nn.ModuleList()
        self.extra_convs = nn.ModuleList()
        self.bifpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            if in_channels[i] == out_channels:
                l_conv = nn.Identity()
            else:
                l_conv = ConvModule(
                    in_channels[i],
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    bias=True,
                    act_cfg=act_cfg,
                    inplace=False)
            self.lateral_convs.append(l_conv)

        for _ in range(repeat_times):
            self.bifpn_convs.append(
                BiFPNLayer(
                    channels=out_channels,
                    levels=num_outs,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    pool_cfg=pool_cfg))

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                if in_channels == out_channels:
                    extra_fpn_conv = nn.MaxPool2d(
                        kernel_size=3, stride=2, padding=1)
                else:
                    extra_fpn_conv = nn.Sequential(
                        ConvModule(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=1,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
                self.extra_convs.append(extra_fpn_conv)

    def forward(self, inputs):

        def extra_convs(inputs, extra_convs):
            outputs = list()
            for extra_conv in extra_convs:
                inputs = extra_conv(inputs)
                outputs.append(inputs)
            return outputs

        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        if self.num_outs > len(laterals) and self.add_extra_convs:
            extra_source = inputs[self.backbone_end_level - 1]
            for extra_conv in self.extra_convs:
                extra_source = extra_conv(extra_source)
                laterals.append(extra_source)

        for bifpn_module in self.bifpn_convs:
            laterals = bifpn_module(laterals)
        outs = laterals

        return tuple(outs)


def swish(x):
    return x * x.sigmoid()


class BiFPNLayer(BaseModule):

    def __init__(self,
                 channels,
                 levels,
                 init=0.5,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=None,
                 pool_cfg=None,
                 eps=0.0001,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.act_cfg = act_cfg
        self.upsample_cfg = upsample_cfg
        self.pool_cfg = pool_cfg
        self.eps = eps
        self.levels = levels
        self.bifpn_convs = nn.ModuleList()
        # weighted
        self.weight_two_nodes = nn.Parameter(
            torch.Tensor(2, levels).fill_(init), requires_grad=True)

        self.weight_three_nodes = nn.Parameter(
            torch.Tensor(3, levels - 2).fill_(init), requires_grad=True)

        # self.relu = nn.ReLU(inplace=False)
        for _ in range(2):
            for _ in range(self.levels - 1):  # 1,2,3
                fpn_conv = nn.Sequential(
                    ConvModule(
                        channels,
                        channels,
                        3,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        inplace=False))
                self.bifpn_convs.append(fpn_conv)

    def forward(self, inputs):
        assert len(inputs) == self.levels
        # build top-down and down-top path with stack
        levels = self.levels
        # w relu

        _w1 = F.relu(self.weight_two_nodes)
        w1 = _w1 / (torch.sum(_w1, dim=0) + self.eps)  # normalize
        w2 = F.relu(self.weight_three_nodes)
        # w2 /= torch.sum(w2, dim=0) + self.eps  # normalize
        # build top-down
        idx_bifpn = 0
        pathtd = inputs
        inputs_clone = []
        for in_tensor in inputs:
            inputs_clone.append(in_tensor.clone())

        for i in range(levels - 1, 0, -1):
            _, _, h, w = pathtd[i - 1].shape
            # pathtd[i - 1] = (
            #     w1[0, i - 1] * pathtd[i - 1] + w1[1, i - 1] *
            #     F.interpolate(pathtd[i], size=(h, w), mode='nearest')) / (
            #         w1[0, i - 1] + w1[1, i - 1] + self.eps)
            pathtd[i -
                   1] = w1[0, i -
                           1] * pathtd[i - 1] + w1[1, i - 1] * F.interpolate(
                               pathtd[i], size=(h, w), mode='nearest')
            pathtd[i - 1] = swish(pathtd[i - 1])
            pathtd[i - 1] = self.bifpn_convs[idx_bifpn](pathtd[i - 1])
            idx_bifpn = idx_bifpn + 1
        # build down-top
        for i in range(0, levels - 2, 1):
            tmp_path = torch.stack([
                inputs_clone[i + 1], pathtd[i + 1],
                F.max_pool2d(pathtd[i], kernel_size=3, stride=2, padding=1)
            ],
                                   dim=-1)
            norm_weight = w2[:, i] / (w2[:, i].sum() + self.eps)
            pathtd[i + 1] = (norm_weight * tmp_path).sum(dim=-1)
            # pathtd[i + 1] = w2[0, i] * inputs_clone[i + 1]
            #     + w2[1, i] * pathtd[
            #     i + 1] + w2[2, i] * F.max_pool2d(
            #         pathtd[i], kernel_size=3, stride=2, padding=1)
            pathtd[i + 1] = swish(pathtd[i + 1])
            pathtd[i + 1] = self.bifpn_convs[idx_bifpn](pathtd[i + 1])
            idx_bifpn = idx_bifpn + 1

        pathtd[levels - 1] = w1[0, levels - 1] * pathtd[levels - 1] + w1[
            1, levels - 1] * F.max_pool2d(
                pathtd[levels - 2], kernel_size=3, stride=2, padding=1)
        pathtd[levels - 1] = swish(pathtd[levels - 1])
        pathtd[levels - 1] = self.bifpn_convs[idx_bifpn](pathtd[levels - 1])
        return pathtd

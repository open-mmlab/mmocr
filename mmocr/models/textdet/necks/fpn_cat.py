import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16

from mmdet.models.builder import NECKS


@NECKS.register_module()
class FPNC(nn.Module):
    """FPN-like fusion module in Real-time Scene Text Detection with
    Differentiable Binarization.

    This was partially adapted from https://github.com/MhLiao/DB and
    https://github.com/WenmuZhou/DBNet.pytorch
    """

    def __init__(self,
                 in_channels,
                 lateral_channels=256,
                 out_channels=64,
                 bias_on_lateral=False,
                 bn_re_on_lateral=False,
                 bias_on_smooth=False,
                 bn_re_on_smooth=False,
                 conv_after_concat=False):
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.lateral_channels = lateral_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.bn_re_on_lateral = bn_re_on_lateral
        self.bn_re_on_smooth = bn_re_on_smooth
        self.conv_after_concat = conv_after_concat
        self.lateral_convs = nn.ModuleList()
        self.smooth_convs = nn.ModuleList()
        self.num_outs = self.num_ins

        for i in range(self.num_ins):
            norm_cfg = None
            act_cfg = None
            if self.bn_re_on_lateral:
                norm_cfg = dict(type='BN')
                act_cfg = dict(type='ReLU')
            l_conv = ConvModule(
                in_channels[i],
                lateral_channels,
                1,
                bias=bias_on_lateral,
                conv_cfg=None,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            norm_cfg = None
            act_cfg = None
            if self.bn_re_on_smooth:
                norm_cfg = dict(type='BN')
                act_cfg = dict(type='ReLU')

            smooth_conv = ConvModule(
                lateral_channels,
                out_channels,
                3,
                bias=bias_on_smooth,
                padding=1,
                conv_cfg=None,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.smooth_convs.append(smooth_conv)
        if self.conv_after_concat:
            norm_cfg = dict(type='BN')
            act_cfg = dict(type='ReLU')
            self.out_conv = ConvModule(
                out_channels * self.num_outs,
                out_channels * self.num_outs,
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.lateral_convs:
            m.init_weights()
        for m in self.smooth_convs:
            m.init_weights()
        if self.conv_after_concat:
            self.out_conv.init_weights()

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        used_backbone_levels = len(laterals)
        # build top-down path
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, mode='nearest')
        # build outputs
        # part 1: from original levels
        outs = [
            self.smooth_convs[i](laterals[i])
            for i in range(used_backbone_levels)
        ]

        for i, out in enumerate(outs):
            outs[i] = F.interpolate(
                outs[i], size=outs[0].shape[2:], mode='nearest')
        out = torch.cat(outs, dim=1)

        if self.conv_after_concat:
            out = self.out_conv(out)

        return out

import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.ops import DeformConv2dPack
from torch import nn

from mmdet.models.builder import NECKS


class CharAttn(nn.Module):
    """Implementation of Character attention module in `CA-FCN.

    <https://arxiv.org/pdf/1809.06508.pdf>`_
    """

    def __init__(self, in_channels=128, out_channels=128, deformable=False):
        super().__init__()
        assert isinstance(in_channels, int)
        assert isinstance(deformable, bool)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deformable = deformable

        # attention layers
        self.attn_layer = nn.Sequential(
            ConvModule(
                in_channels,
                in_channels,
                3,
                stride=1,
                padding=1,
                norm_cfg=dict(type='BN')),
            ConvModule(
                in_channels,
                1,
                3,
                stride=1,
                padding=1,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='Sigmoid')))

        conv_kwargs = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=1,
            padding=1)
        if self.deformable:
            self.conv = DeformConv2dPack(**conv_kwargs)
        else:
            self.conv = nn.Conv2d(**conv_kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, in_feat):
        # Calculate attn map
        attn_map = self.attn_layer(in_feat)  # N * 1 * H * W

        in_feat = self.relu(self.bn(self.conv(in_feat)))

        out_feat_map = self._upsample_mul(in_feat, 1 + attn_map)

        return out_feat_map, attn_map

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:]) + y

    def _upsample_mul(self, x, y):
        return F.interpolate(x, size=y.size()[2:]) * y


class FeatGenerator(nn.Module):
    """Generate attention-augmented stage feature from backbone stage
    feature."""

    def __init__(self,
                 in_channels=512,
                 out_channels=128,
                 deformable=True,
                 concat=False,
                 upsample=False,
                 with_attn=True):
        super().__init__()

        self.concat = concat
        self.upsample = upsample
        self.with_attn = with_attn

        if with_attn:
            self.char_attn = CharAttn(
                in_channels=in_channels,
                out_channels=out_channels,
                deformable=deformable)
        else:
            self.char_attn = ConvModule(
                in_channels,
                out_channels,
                3,
                stride=1,
                padding=1,
                norm_cfg=dict(type='BN'))

        if concat:
            self.conv_to_concat = ConvModule(
                out_channels,
                out_channels,
                3,
                stride=1,
                padding=1,
                norm_cfg=dict(type='BN'))

        kernel_size = (3, 1) if deformable else 3
        padding = (1, 0) if deformable else 1
        tmp_in_channels = out_channels * 2 if concat else out_channels

        self.conv_after_concat = ConvModule(
            tmp_in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            norm_cfg=dict(type='BN'))

    def forward(self, x, y=None, size=None):
        if self.with_attn:
            feat_map, attn_map = self.char_attn(x)
        else:
            feat_map = self.char_attn(x)
            attn_map = feat_map

        if self.concat:
            y = self.conv_to_concat(y)
            feat_map = torch.cat((y, feat_map), dim=1)

        feat_map = self.conv_after_concat(feat_map)

        if self.upsample:
            feat_map = F.interpolate(feat_map, size)

        return attn_map, feat_map


@NECKS.register_module()
class CAFCNNeck(nn.Module):
    """Implementation of neck module in `CA-FCN.

    <https://arxiv.org/pdf/1809.06508.pdf>`_

    Args:
        in_channels (list[int]): Number of input channels for each scale.
        out_channels (int): Number of output channels for each scale.
        deformable (bool): If True, use deformable conv.
        with_attn (bool): If True, add attention for each output feature map.
    """

    def __init__(self,
                 in_channels=[128, 256, 512, 512],
                 out_channels=128,
                 deformable=True,
                 with_attn=True):
        super().__init__()

        self.deformable = deformable
        self.with_attn = with_attn

        # stage_in5_to_out5
        self.s5 = FeatGenerator(
            in_channels=in_channels[-1],
            out_channels=out_channels,
            deformable=deformable,
            concat=False,
            with_attn=with_attn)

        # stage_in4_to_out4
        self.s4 = FeatGenerator(
            in_channels=in_channels[-2],
            out_channels=out_channels,
            deformable=deformable,
            concat=True,
            with_attn=with_attn)

        # stage_in3_to_out3
        self.s3 = FeatGenerator(
            in_channels=in_channels[-3],
            out_channels=out_channels,
            deformable=False,
            concat=True,
            upsample=True,
            with_attn=with_attn)

        # stage_in2_to_out2
        self.s2 = FeatGenerator(
            in_channels=in_channels[-4],
            out_channels=out_channels,
            deformable=False,
            concat=True,
            upsample=True,
            with_attn=with_attn)

    def init_weights(self):
        pass

    def forward(self, feats):
        in_stage1 = feats[0]
        in_stage2, in_stage3 = feats[1], feats[2]
        in_stage4, in_stage5 = feats[3], feats[4]
        # out stage 5
        out_s5_attn_map, out_s5 = self.s5(in_stage5)

        # out stage 4
        out_s4_attn_map, out_s4 = self.s4(in_stage4, out_s5)

        # out stage 3
        out_s3_attn_map, out_s3 = self.s3(in_stage3, out_s4,
                                          in_stage2.size()[2:])

        # out stage 2
        out_s2_attn_map, out_s2 = self.s2(in_stage2, out_s3,
                                          in_stage1.size()[2:])

        return (out_s2_attn_map, out_s3_attn_map, out_s4_attn_map,
                out_s5_attn_map, out_s2)

import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, ModuleList

from mmocr.models.builder import NECKS


@NECKS.register_module()
class FPNOCR(BaseModule):
    """FPN-like Network for segmentation based text recognition.

    Args:
        in_channels (list[int]): Number of input channels for each scale.
        out_channels (int): Number of output channels for each scale.
        last_stage_only (bool): If True, output last stage only.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 last_stage_only=True,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)

        self.last_stage_only = last_stage_only

        self.lateral_convs = ModuleList()
        self.smooth_convs_1x1 = ModuleList()
        self.smooth_convs_3x3 = ModuleList()

        for i in range(self.num_ins):
            l_conv = ConvModule(
                in_channels[i], out_channels, 1, norm_cfg=dict(type='BN'))
            self.lateral_convs.append(l_conv)

        for i in range(self.num_ins - 1):
            s_conv_1x1 = ConvModule(
                out_channels * 2, out_channels, 1, norm_cfg=dict(type='BN'))
            s_conv_3x3 = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                norm_cfg=dict(type='BN'))
            self.smooth_convs_1x1.append(s_conv_1x1)
            self.smooth_convs_3x3.append(s_conv_3x3)

    def _upsample_x2(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear')

    def forward(self, inputs):
        lateral_features = [
            l_conv(inputs[i]) for i, l_conv in enumerate(self.lateral_convs)
        ]

        outs = []
        for i in range(len(self.smooth_convs_3x3), 0, -1):  # 3, 2, 1
            last_out = lateral_features[-1] if len(outs) == 0 else outs[-1]
            upsample = self._upsample_x2(last_out)
            upsample_cat = torch.cat((upsample, lateral_features[i - 1]),
                                     dim=1)
            smooth_1x1 = self.smooth_convs_1x1[i - 1](upsample_cat)
            smooth_3x3 = self.smooth_convs_3x3[i - 1](smooth_1x1)
            outs.append(smooth_3x3)

        return tuple(outs[-1:]) if self.last_stage_only else tuple(outs)

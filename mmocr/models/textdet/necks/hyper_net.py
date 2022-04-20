# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, ModuleList, Sequential

from mmocr.models.builder import NECKS


@NECKS.register_module()
class HyperNet(BaseModule):

    def __init__(self, in_channels, stage_out_channels, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        assert isinstance(stage_out_channels, list)
        # see the last in channel as the first stage out channel
        self.in_channels = in_channels[:-1]
        self.stage_out_channels = [in_channels[-1]] + stage_out_channels
        self.num_stages = len(self.in_channels)
        self.convstages = ModuleList()
        for i in range(self.num_stages):
            convstage = []
            convstage.append(
                ConvModule(
                    self.in_channels[self.num_stages - i - 1] +
                    self.stage_out_channels[i],
                    self.stage_out_channels[i + 1],
                    kernel_size=1,
                    stride=1,
                    bias=False,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='ReLU')))
            convstage.append(
                ConvModule(
                    self.stage_out_channels[i + 1],
                    self.stage_out_channels[i + 1],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='ReLU')))
            self.convstages.append(Sequential(*convstage))

        self.out_conv = ConvModule(
            stage_out_channels[-1],
            stage_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels) + 1
        out = inputs[-1]
        for i in range(self.num_stages):
            input_idx = self.num_stages - i
            prev_shape = inputs[input_idx - 1].shape[2:]
            out = torch.cat([
                inputs[input_idx - 1],
                F.interpolate(out, size=prev_shape, mode='nearest')
            ],
                            dim=1)
            out = self.convstages[i](out)
        return self.out_conv(out)

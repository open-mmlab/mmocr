# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch

from mmocr.models.textdet.necks import FPNC


class TestFPNC(unittest.TestCase):

    def test_forward(self):
        in_channels = [64, 128, 256, 512]
        size = [112, 56, 28, 14]
        asf_cfgs = [
            None,
            dict(attention_type='ScaleChannelSpatial'),
        ]
        for flag in [False, True]:
            for asf_cfg in asf_cfgs:
                fpnc = FPNC(
                    in_channels=in_channels,
                    bias_on_lateral=flag,
                    bn_re_on_lateral=flag,
                    bias_on_smooth=flag,
                    bn_re_on_smooth=flag,
                    asf_cfg=asf_cfg,
                    conv_after_concat=flag)
            fpnc.init_weights()
            inputs = []
            for i in range(4):
                inputs.append(torch.rand(1, in_channels[i], size[i], size[i]))
            outputs = fpnc.forward(inputs)
            self.assertListEqual(list(outputs.size()), [1, 256, 112, 112])

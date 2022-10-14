# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer

from mmocr.models.common.backbones import CLIPResNet
from mmocr.models.common.backbones.clip_resnet import CLIPBottleneck


class TestCLIPResNet(TestCase):

    def test_forward(self):
        model = CLIPResNet()
        model.eval()

        imgs = torch.randn(1, 3, 32, 32)
        feat = model(imgs)
        assert len(feat) == 4
        assert feat[0].shape == torch.Size([1, 256, 8, 8])
        assert feat[1].shape == torch.Size([1, 512, 4, 4])
        assert feat[2].shape == torch.Size([1, 1024, 2, 2])
        assert feat[3].shape == torch.Size([1, 2048, 1, 1])


class TestCLIPBottleneck(TestCase):

    def test_forward(self):
        stride = 1
        inplanes = 64
        planes = 64
        conv_cfg = None
        norm_cfg = {'type': 'BN', 'requires_grad': True}

        downsample = []
        downsample.append(
            nn.AvgPool2d(
                kernel_size=stride,
                stride=stride,
                ceil_mode=True,
                count_include_pad=False))
        downsample.extend([
            build_conv_layer(
                conv_cfg,
                inplanes,
                planes * CLIPBottleneck.expansion,
                kernel_size=1,
                stride=stride,
                bias=False),
            build_norm_layer(norm_cfg, planes * CLIPBottleneck.expansion)[1]
        ])
        downsample = nn.Sequential(*downsample)

        model = CLIPBottleneck(
            inplanes=64,
            planes=64,
            stride=stride,
            downsample=downsample,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        model.eval()

        input_feat = torch.randn(1, 64, 8, 8)
        output_feat = model(input_feat)
        assert output_feat.shape == torch.Size([1, 256, 8, 8])

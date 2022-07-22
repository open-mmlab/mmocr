# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmocr.models.textrecog.backbones import ResNet


class TestResNet(TestCase):

    def setUp(self) -> None:
        self.img = torch.rand(1, 3, 32, 100)

    def test_resnet45_aster(self):
        resnet45_aster = ResNet(
            in_channels=3,
            stem_channels=[64, 128],
            block_cfgs=dict(type='BasicBlock', use_conv1x1='True'),
            arch_layers=[3, 4, 6, 6, 3],
            arch_channels=[32, 64, 128, 256, 512],
            strides=[(2, 2), (2, 2), (2, 1), (2, 1), (2, 1)])
        self.assertEqual(
            resnet45_aster(self.img).shape, torch.Size([1, 512, 1, 25]))

    def test_resnet45_abi(self):
        resnet45_abi = ResNet(
            in_channels=3,
            stem_channels=32,
            block_cfgs=dict(type='BasicBlock', use_conv1x1='True'),
            arch_layers=[3, 4, 6, 6, 3],
            arch_channels=[32, 64, 128, 256, 512],
            strides=[2, 1, 2, 1, 1])
        self.assertEqual(
            resnet45_abi(self.img).shape, torch.Size([1, 512, 8, 25]))

    def test_resnet31_master(self):
        resnet31_master = ResNet(
            in_channels=3,
            stem_channels=[64, 128],
            block_cfgs=dict(type='BasicBlock'),
            arch_layers=[1, 2, 5, 3],
            arch_channels=[256, 256, 512, 512],
            strides=[1, 1, 1, 1],
            plugins=[
                dict(
                    cfg=dict(type='Maxpool2d', kernel_size=2, stride=(2, 2)),
                    stages=(True, True, False, False),
                    position='before_stage'),
                dict(
                    cfg=dict(
                        type='Maxpool2d', kernel_size=(2, 1), stride=(2, 1)),
                    stages=(False, False, True, False),
                    position='before_stage'),
                dict(
                    cfg=dict(type='GCAModule', ratio=0.0625, n_head=1),
                    stages=[True, True, True, True],
                    position='after_stage'),
                dict(
                    cfg=dict(
                        type='ConvModule',
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=dict(type='BN'),
                        act_cfg=dict(type='ReLU')),
                    stages=(True, True, True, True),
                    position='after_stage')
            ])
        self.assertEqual(
            resnet31_master(self.img).shape, torch.Size([1, 512, 4, 25]))

    def test_resnet31(self):
        resnet_31 = ResNet(
            in_channels=3,
            stem_channels=[64, 128],
            block_cfgs=dict(type='BasicBlock'),
            arch_layers=[1, 2, 5, 3],
            arch_channels=[256, 256, 512, 512],
            strides=[1, 1, 1, 1],
            plugins=[
                dict(
                    cfg=dict(type='Maxpool2d', kernel_size=2, stride=(2, 2)),
                    stages=(True, True, False, False),
                    position='before_stage'),
                dict(
                    cfg=dict(
                        type='Maxpool2d', kernel_size=(2, 1), stride=(2, 1)),
                    stages=(False, False, True, False),
                    position='before_stage'),
                dict(
                    cfg=dict(
                        type='ConvModule',
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=dict(type='BN'),
                        act_cfg=dict(type='ReLU')),
                    stages=(True, True, True, True),
                    position='after_stage')
            ])
        self.assertEqual(
            resnet_31(self.img).shape, torch.Size([1, 512, 4, 25]))

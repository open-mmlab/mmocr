# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmocr.models.textrecog.layers.conv_layer import (BasicBlock, Bottleneck,
                                                      conv1x1, conv3x3)


class TestUtils(TestCase):

    def test_conv3x3(self):
        conv = conv3x3(3, 6)
        self.assertEqual(conv.in_channels, 3)
        self.assertEqual(conv.out_channels, 6)
        self.assertEqual(conv.kernel_size, (3, 3))

    def test_conv1x1(self):
        conv = conv1x1(3, 6)
        self.assertEqual(conv.in_channels, 3)
        self.assertEqual(conv.out_channels, 6)
        self.assertEqual(conv.kernel_size, (1, 1))


class TestBasicBlock(TestCase):

    def test_forward(self):
        x = torch.rand(1, 64, 224, 224)
        basic_block = BasicBlock(64, 64)
        self.assertEqual(basic_block.expansion, 1)
        out = basic_block(x)
        self.assertEqual(out.shape, torch.Size([1, 64, 224, 224]))


class TestBottleneck(TestCase):

    def test_forward(self):
        x = torch.rand(1, 64, 224, 224)
        bottle_neck = Bottleneck(64, 64, downsample=True)
        self.assertEqual(bottle_neck.expansion, 4)
        out = bottle_neck(x)
        self.assertEqual(out.shape, torch.Size([1, 256, 224, 224]))

# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch

from mmocr.models.textdet.necks import FPN_UNet


class TestFPNUnet(unittest.TestCase):

    def setUp(self):
        self.s = 64
        feat_sizes = [self.s // 2**i for i in range(4)]
        self.in_channels = [8, 16, 32, 64]
        self.out_channels = 4
        self.feature = [
            torch.rand(1, self.in_channels[i], feat_sizes[i], feat_sizes[i])
            for i in range(len(self.in_channels))
        ]

    def test_init(self):
        with self.assertRaises(AssertionError):
            FPN_UNet(self.in_channels + [128], self.out_channels)
        with self.assertRaises(AssertionError):
            FPN_UNet(self.in_channels, [2, 4])

    def test_forward(self):
        neck = FPN_UNet(self.in_channels, self.out_channels)
        neck.init_weights()
        out = neck(self.feature)
        self.assertTrue(out.shape == torch.Size(
            [1, self.out_channels, self.s * 4, self.s * 4]))

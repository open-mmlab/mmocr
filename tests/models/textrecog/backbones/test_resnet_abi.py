# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmocr.models.textrecog.backbones import ResNetABI


class TestResNetABI(TestCase):

    def test_forward(self):
        """Test resnet backbone."""
        with self.assertRaises(AssertionError):
            ResNetABI(2.5)

        with self.assertRaises(AssertionError):
            ResNetABI(3, arch_settings=5)

        with self.assertRaises(AssertionError):
            ResNetABI(3, stem_channels=None)

        with self.assertRaises(AssertionError):
            ResNetABI(arch_settings=[3, 4, 6, 6], strides=[1, 2, 1, 2, 1])

        # Test forwarding
        model = ResNetABI()
        model.train()

        imgs = torch.randn(1, 3, 32, 160)
        feat = model(imgs)
        self.assertEqual(feat.shape, torch.Size([1, 512, 8, 40]))

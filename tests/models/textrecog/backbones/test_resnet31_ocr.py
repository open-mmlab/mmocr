# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmocr.models.textrecog.backbones import ResNet31OCR


class TestResNet31OCR(TestCase):

    def test_forward(self):
        """Test resnet backbone."""
        with self.assertRaises(AssertionError):
            ResNet31OCR(2.5)

        with self.assertRaises(AssertionError):
            ResNet31OCR(3, layers=5)

        with self.assertRaises(AssertionError):
            ResNet31OCR(3, channels=5)

        # Test ResNet18 forward
        model = ResNet31OCR()
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 32, 160)
        feat = model(imgs)
        self.assertEqual(feat.shape, torch.Size([1, 512, 4, 40]))

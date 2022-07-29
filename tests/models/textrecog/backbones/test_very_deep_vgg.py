# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmocr.models.textrecog.backbones import VeryDeepVgg


class TestVeryDeepVgg(TestCase):

    def test_forward(self):

        model = VeryDeepVgg()
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 32, 160)
        feats = model(imgs)
        self.assertEqual(feats.shape, torch.Size([1, 512, 1, 41]))

# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch

from mmocr.models.textrecog.backbones import ShallowCNN


class TestShallowCNN(unittest.TestCase):

    def setUp(self):
        self.imgs = torch.randn(1, 1, 32, 100)

    def test_shallow_cnn(self):

        model = ShallowCNN()
        model.init_weights()
        model.train()

        feat = model(self.imgs)
        self.assertEqual(feat.shape, torch.Size([1, 512, 8, 25]))

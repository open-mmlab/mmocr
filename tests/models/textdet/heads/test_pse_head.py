# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmocr.models.textdet.heads import PSEHead


class TestPSEHead(TestCase):

    def setUp(self):
        self.feature = torch.randn((2, 10, 40, 50))

    def test_init(self):
        with self.assertRaises(TypeError):
            PSEHead(in_channels=1)

        with self.assertRaises(TypeError):
            PSEHead(out_channels='out')

    def test_forward(self):
        pse_head = PSEHead(in_channels=[10], hidden_dim=128, out_channel=7)
        results = pse_head(self.feature)
        self.assertEqual(results.shape, (2, 7, 40, 50))

# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmocr.models.textdet.heads import PANHead


class TestPANHead(TestCase):

    def test_init(self):
        with self.assertRaises(AssertionError):
            PANHead(in_channels='test', hidden_dim=128, out_channel=6)
        with self.assertRaises(AssertionError):
            PANHead(in_channels=['test'], hidden_dim=128, out_channel=6)
        with self.assertRaises(AssertionError):
            PANHead(in_channels=[128, 128], hidden_dim='test', out_channel=6)
        with self.assertRaises(AssertionError):
            PANHead(in_channels=[128, 128], hidden_dim=128, out_channel='test')

    def test_forward(self):
        pan_head = PANHead(in_channels=[10], hidden_dim=128, out_channel=6)
        data = torch.randn((2, 10, 40, 50))
        results = pan_head(data)
        self.assertEqual(results.shape, (2, 6, 40, 50))

# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from parameterized import parameterized

from mmocr.models.textrecog.plugins import GCAModule


class TestGCAModule(TestCase):

    def setUp(self) -> None:
        self.img = torch.rand(1, 32, 32, 100)

    @parameterized.expand([('att'), ('avg')])
    def test_gca_module_pooling(self, pooling_type):
        gca_module = GCAModule(
            in_channels=32,
            ratio=0.0625,
            n_head=1,
            pooling_type=pooling_type,
            is_att_scale=False,
            fusion_type='channel_add')
        self.assertEqual(
            gca_module(self.img).shape, torch.Size([1, 32, 32, 100]))

    @parameterized.expand([('channel_add'), ('channel_mul'),
                           ('channel_concat')])
    def test_gca_module_fusion(self, fusion_type):
        gca_module = GCAModule(
            in_channels=32,
            ratio=0.0625,
            n_head=1,
            pooling_type='att',
            is_att_scale=False,
            fusion_type=fusion_type)
        self.assertEqual(
            gca_module(self.img).shape, torch.Size([1, 32, 32, 100]))

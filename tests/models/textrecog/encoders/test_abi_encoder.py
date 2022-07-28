# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmocr.models.textrecog.encoders.abi_encoder import ABIEncoder


class TestABIEncoder(TestCase):

    def test_init(self):
        with self.assertRaises(AssertionError):
            ABIEncoder(d_model=512, n_head=10)

    def test_forward(self):
        model = ABIEncoder()
        x = torch.randn(10, 512, 8, 32)
        self.assertEqual(model(x, None).shape, torch.Size([10, 512, 8, 32]))

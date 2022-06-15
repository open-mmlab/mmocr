# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmocr.models.textdet.heads import TextSnakeHead


class TestTextSnakeHead(TestCase):

    def test_init(self):
        with self.assertRaises(AssertionError):
            TextSnakeHead(in_channels='test')

    def test_forward(self):
        ts_head = TextSnakeHead(in_channels=10)
        data = torch.randn((2, 10, 40, 50))
        results = ts_head(data, None)
        self.assertEqual(results.shape, (2, 5, 40, 50))

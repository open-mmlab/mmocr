# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmocr.models.textdet.heads import FCEHead


class TestFCEHead(TestCase):

    def test_init(self):
        with self.assertRaises(AssertionError):
            FCEHead(in_channels='test', fourier_degree=5)

        with self.assertRaises(AssertionError):
            FCEHead(in_channels=1, fourier_degree='Text')

    def test_forward(self):
        fce_head = FCEHead(in_channels=10, fourier_degree=5)
        data = [
            torch.randn(2, 10, 20, 20),
            torch.randn(2, 10, 30, 30),
            torch.randn(2, 10, 40, 40)
        ]
        results = fce_head(data)
        self.assertIn('cls_res', results[0])
        self.assertIn('reg_res', results[0])
        self.assertEqual(results[0]['cls_res'].shape, (2, 4, 20, 20))
        self.assertEqual(results[0]['reg_res'].shape, (2, 22, 20, 20))
        self.assertEqual(results[1]['cls_res'].shape, (2, 4, 30, 30))
        self.assertEqual(results[1]['reg_res'].shape, (2, 22, 30, 30))
        self.assertEqual(results[2]['cls_res'].shape, (2, 4, 40, 40))
        self.assertEqual(results[2]['reg_res'].shape, (2, 22, 40, 40))

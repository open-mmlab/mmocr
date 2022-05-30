# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmocr.models.textdet.heads import DBHead


class TestDBHead(TestCase):

    def test_init(self):
        with self.assertRaises(AssertionError):
            DBHead(in_channels='test', with_bias=False)

        with self.assertRaises(AssertionError):
            DBHead(in_channels=1, with_bias='Text')

    def test_forward(self):
        db_head = DBHead(in_channels=10)
        data = torch.randn((2, 10, 40, 50))
        results = db_head(data, None)
        self.assertIn('prob_map', results)
        self.assertIn('thr_map', results)
        self.assertIn('binary_map', results)
        self.assertEqual(results['prob_map'].shape, (2, 160, 200))
        self.assertEqual(results['thr_map'].shape, (2, 160, 200))
        self.assertEqual(results['binary_map'].shape, (2, 160, 200))

# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmocr.models.textdet.heads import DBHead
from mmocr.registry import MODELS


class TestDBHead(TestCase):

    # Use to replace module loss and postprocessors
    @MODELS.register_module(name='DBDummy')
    class DummyModule:

        def __call__(self, x, data_samples):
            return x

    def setUp(self) -> None:
        self.db_head = DBHead(
            in_channels=10,
            module_loss=dict(type='DBDummy'),
            postprocessor=dict(type='DBDummy'))

    def test_init(self):
        with self.assertRaises(AssertionError):
            DBHead(in_channels='test', with_bias=False)

        with self.assertRaises(AssertionError):
            DBHead(in_channels=1, with_bias='Text')

    def test_forward(self):
        data = torch.randn((2, 10, 40, 50))

        results = self.db_head(data, None, 'loss')
        for i in range(3):
            self.assertEqual(results[i].shape, (2, 160, 200))

        results = self.db_head(data, None, 'predict')
        self.assertEqual(results.shape, (2, 160, 200))

        results = self.db_head(data, None, 'both')
        for i in range(4):
            self.assertEqual(results[i].shape, (2, 160, 200))
        self.assertTrue(torch.allclose(results[3], results[0].sigmoid()))

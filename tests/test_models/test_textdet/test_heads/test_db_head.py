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
        results = self.db_head(data, None)
        self.assertEqual(results[0].shape, (2, 160, 200))
        self.assertEqual(results[1].shape, (2, 160, 200))
        self.assertEqual(results[2].shape, (2, 160, 200))

    def test_loss(self):
        data = torch.randn((2, 10, 40, 50))
        results = self.db_head.loss(data, None)
        for i in range(3):
            self.assertEqual(results[i].shape, (2, 160, 200))

    def test_predict(self):
        data = torch.randn((2, 10, 40, 50))
        results = self.db_head.predict(data, None)
        self.assertEqual(results.shape, (2, 160, 200))

    def test_loss_and_predict(self):
        data = torch.randn((2, 10, 40, 50))
        loss_results, pred_results = self.db_head.loss_and_predict(data, None)
        for i in range(3):
            self.assertEqual(loss_results[i].shape, (2, 160, 200))
        self.assertEqual(pred_results.shape, (2, 160, 200))
        self.assertTrue(
            torch.allclose(pred_results, loss_results[0].sigmoid()))

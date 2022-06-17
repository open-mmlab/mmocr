# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch
import torch.nn as nn
from mmengine import InstanceData

from mmocr.core import TextDetDataSample
from mmocr.models.textdet.losses import PANLoss
from mmocr.models.textdet.losses.pan_loss import PANEmbLossV1


class TestPANLoss(TestCase):

    def setUp(self) -> None:

        self.data_samples = [
            TextDetDataSample(
                metainfo=dict(img_shape=(40, 40)),
                gt_instances=InstanceData(
                    polygons=np.array([
                        [0, 0, 10, 0, 10, 10, 0, 10],
                        [20, 0, 30, 0, 30, 10, 20, 10],
                        [0, 0, 15, 0, 15, 10, 0, 10],
                    ],
                                      dtype=np.float32),
                    ignored=torch.BoolTensor([False, False, True])))
        ]
        pred_size = (1, 6, 10, 10)
        self.preds = torch.rand(pred_size)

    def test_init(self):
        with self.assertRaises(AssertionError):
            PANLoss(reduction=1)
        pan_loss = PANLoss()
        self.assertIsInstance(pan_loss.loss_text, nn.Module)
        self.assertIsInstance(pan_loss.loss_kernel, nn.Module)
        self.assertIsInstance(pan_loss.loss_embedding, nn.Module)

    def test_get_target(self):
        pan_loss = PANLoss()
        gt_kernels, gt_masks = pan_loss.get_targets(self.data_samples)
        self.assertEqual(gt_kernels.shape, (2, 1, 40, 40))
        self.assertEqual(gt_masks.shape, (1, 40, 40))

    def test_pan_loss(self):
        pan_loss = PANLoss()
        loss = pan_loss(self.preds, self.data_samples)
        self.assertIn('loss_text', loss)
        self.assertIn('loss_kernel', loss)
        self.assertIn('loss_embedding', loss)


class TestPANEmbLossV1(TestCase):

    def test_forward(self):
        loss = PANEmbLossV1()

        pred = torch.rand((2, 4, 10, 10))
        gt = torch.rand((2, 10, 10))
        mask = torch.rand((2, 10, 10))
        instance = torch.zeros_like(gt)
        instance[:, 2:4, 2:4] = 1
        instance[:, 6:8, 6:8] = 2

        loss_value = loss(pred, instance, gt, mask)
        self.assertEqual(loss_value.shape, torch.Size([2]))

        instance = instance = torch.zeros_like(gt)
        loss_value = loss(pred, instance, gt, mask)
        self.assertTrue((loss_value == torch.zeros(2,
                                                   dtype=torch.float32)).all())

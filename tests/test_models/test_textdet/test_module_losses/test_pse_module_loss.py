# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch
import torch.nn as nn
from mmengine.structures import InstanceData
from parameterized import parameterized

from mmocr.models.textdet.module_losses import PSEModuleLoss
from mmocr.structures import TextDetDataSample


class TestPSEModuleLoss(TestCase):

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
        pred_size = (1, 7, 10, 10)
        self.preds = torch.rand(pred_size)

    def test_init(self):
        with self.assertRaises(AssertionError):
            PSEModuleLoss(reduction=1)
        pse_loss = PSEModuleLoss(reduction='sum')
        self.assertIsInstance(pse_loss.loss_text, nn.Module)
        self.assertIsInstance(pse_loss.loss_kernel, nn.Module)

    @parameterized.expand([('mean', 'hard'), ('sum', 'adaptive')])
    def test_forward(self, reduction, kernel_sample_type):
        pse_loss = PSEModuleLoss(
            reduction=reduction, kernel_sample_type=kernel_sample_type)
        loss = pse_loss(self.preds, self.data_samples)
        self.assertIn('loss_text', loss)
        self.assertIn('loss_kernel', loss)

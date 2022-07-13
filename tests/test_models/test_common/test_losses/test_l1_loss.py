# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmocr.models.common.losses import MaskedSmoothL1Loss


class TestMaskedSmoothL1Loss(TestCase):

    def setUp(self) -> None:
        self.l1_loss = MaskedSmoothL1Loss(beta=0)
        self.smooth_l1_loss = MaskedSmoothL1Loss(beta=1)
        self.pred = torch.FloatTensor([0.5, 1, 1.5, 2])
        self.gt = torch.ones_like(self.pred)
        self.mask = torch.FloatTensor([1, 0, 0, 1])

    def test_forward(self):

        # Shape mismatch between pred and gt
        with self.assertRaises(AssertionError):
            invalid_gt = torch.FloatTensor([0, 0, 0])
            self.l1_loss(self.pred, invalid_gt)

        # Shape mismatch between pred and mask
        with self.assertRaises(AssertionError):
            invalid_mask = torch.BoolTensor([True, False, False])
            self.l1_loss(self.pred, self.gt, invalid_mask)

        # Test L1 loss results
        self.assertAlmostEqual(
            self.l1_loss(self.pred, self.gt).item(), 0.5, delta=0.01)
        self.assertAlmostEqual(
            self.l1_loss(self.pred, self.gt, self.mask).item(),
            0.75,
            delta=0.01)

        # Test Smooth L1 loss results
        self.assertAlmostEqual(
            self.smooth_l1_loss(self.pred, self.gt, self.mask).item(),
            0.3125,
            delta=0.01)

        # Test zero mask
        zero_mask = torch.FloatTensor([0, 0, 0, 0])
        self.assertAlmostEqual(
            self.smooth_l1_loss(self.pred, self.gt, zero_mask).item(), 0)

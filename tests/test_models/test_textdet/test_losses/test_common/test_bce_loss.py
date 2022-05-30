# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmocr.models.textdet.losses.common import MaskedBalancedBCELoss


class TestMaskedBalancedBCELoss(TestCase):

    def setUp(self) -> None:
        self.bce_loss = MaskedBalancedBCELoss(negative_ratio=2)
        self.pred = torch.FloatTensor([0.1, 0.2, 0.3, 0.4])
        self.gt = torch.FloatTensor([1, 0, 0, 0])
        self.mask = torch.BoolTensor([True, False, False, True])

    def test_init(self):
        with self.assertRaises(AssertionError):
            MaskedBalancedBCELoss(reduction='any')

        with self.assertRaises(AssertionError):
            MaskedBalancedBCELoss(negative_ratio='a')

        with self.assertRaises(AssertionError):
            MaskedBalancedBCELoss(eps='a')

    def test_forward(self):

        # Shape mismatch between pred and gt
        with self.assertRaises(AssertionError):
            invalid_gt = torch.FloatTensor([0, 0, 0])
            self.bce_loss(self.pred, invalid_gt)

        # Shape mismatch between pred and mask
        with self.assertRaises(AssertionError):
            invalid_mask = torch.BoolTensor([True, False, False])
            self.bce_loss(self.pred, self.gt, invalid_mask)

        # Invalid pred or gt
        with self.assertRaises(AssertionError):
            invalid_gt = torch.FloatTensor([2, 3, 4, 5])
            self.bce_loss(self.pred, invalid_gt, self.mask)
        with self.assertRaises(AssertionError):
            invalid_pred = torch.FloatTensor([2, 3, 4, 5])
            self.bce_loss(invalid_pred, self.gt, self.mask)

        self.assertAlmostEqual(
            self.bce_loss(self.pred, self.gt).item(), 1.0567, delta=0.1)
        self.assertAlmostEqual(
            self.bce_loss(self.pred, self.gt, self.mask).item(),
            1.4067,
            delta=0.1)

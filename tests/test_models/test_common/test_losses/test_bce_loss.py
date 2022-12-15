# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmocr.models.common.losses import (MaskedBalancedBCELoss,
                                        MaskedBalancedBCEWithLogitsLoss,
                                        MaskedBCELoss, MaskedBCEWithLogitsLoss)


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

        with self.assertRaises(AssertionError):
            MaskedBalancedBCELoss(fallback_negative_num='a')

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

        # Test zero mask
        zero_mask = torch.FloatTensor([0, 0, 0, 0])
        self.assertAlmostEqual(
            self.bce_loss(self.pred, self.gt, zero_mask).item(), 0)

        # Test 0 < fallback_negative_num < negative numbers
        all_neg_gt = torch.zeros((4, ))
        self.fallback_bce_loss = MaskedBalancedBCELoss(fallback_negative_num=1)
        self.assertAlmostEqual(
            self.fallback_bce_loss(self.pred, all_neg_gt, self.mask).item(),
            0.51,
            delta=0.001)
        # Test fallback_negative_num > negative numbers
        self.fallback_bce_loss = MaskedBalancedBCELoss(fallback_negative_num=3)
        self.assertAlmostEqual(
            self.fallback_bce_loss(self.pred, all_neg_gt, self.mask).item(),
            0.308,
            delta=0.001)


class TestMaskedBCELoss(TestCase):

    def setUp(self) -> None:
        self.bce_loss = MaskedBCELoss()
        self.pred = torch.FloatTensor([0.1, 0.2, 0.3, 0.4])
        self.gt = torch.FloatTensor([1, 0, 0, 0])
        self.mask = torch.BoolTensor([True, False, False, True])

    def test_init(self):
        with self.assertRaises(AssertionError):
            MaskedBCELoss(eps='a')

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
            self.bce_loss(self.pred, self.gt).item(), 0.8483, delta=0.1)
        self.assertAlmostEqual(
            self.bce_loss(self.pred, self.gt, self.mask).item(),
            1.4067,
            delta=0.1)

        # Test zero mask
        zero_mask = torch.FloatTensor([0, 0, 0, 0])
        self.assertAlmostEqual(
            self.bce_loss(self.pred, self.gt, zero_mask).item(), 0)


class TestMaskedBalancedWithLogitsBCELoss(TestCase):

    def setUp(self) -> None:
        self.loss = MaskedBalancedBCEWithLogitsLoss(negative_ratio=2)
        self.pred = torch.FloatTensor([1.5, 1.5, 1.5, 1.5])
        self.gt = torch.FloatTensor([1, 1, 1, 0])
        self.mask = torch.BoolTensor([True, False, False, True])

    def test_init(self):
        with self.assertRaises(AssertionError):
            MaskedBalancedBCEWithLogitsLoss(reduction='any')

        with self.assertRaises(AssertionError):
            MaskedBalancedBCEWithLogitsLoss(negative_ratio='a')

        with self.assertRaises(AssertionError):
            MaskedBalancedBCEWithLogitsLoss(eps='a')

        with self.assertRaises(AssertionError):
            MaskedBalancedBCEWithLogitsLoss(fallback_negative_num='a')

    def test_forward(self):

        # Shape mismatch between pred and gt
        with self.assertRaises(AssertionError):
            invalid_gt = torch.FloatTensor([0, 0, 0])
            self.loss(self.pred, invalid_gt)

        # Shape mismatch between pred and mask
        with self.assertRaises(AssertionError):
            invalid_mask = torch.BoolTensor([True, False, False])
            self.loss(self.pred, self.gt, invalid_mask)

        # Invalid gt
        with self.assertRaises(AssertionError):
            invalid_gt = torch.FloatTensor([2, 3, 4, 5])
            self.loss(self.pred, invalid_gt, self.mask)

        logit = torch.FloatTensor([1.5])
        self.assertAlmostEqual(
            self.loss(self.pred, self.gt).item(),
            ((-torch.log(torch.sigmoid(logit)) * 3 -
              torch.log(1 - torch.sigmoid(logit))) / 4).item(),
            delta=0.0001)
        self.assertAlmostEqual(
            self.loss(self.pred, self.gt, self.mask).item(),
            (-torch.log(torch.sigmoid(logit)) -
             torch.log(1 - torch.sigmoid(logit))).item() / 2,
            delta=0.0001)

        # Test zero mask
        zero_mask = torch.FloatTensor([0, 0, 0, 0])
        self.assertAlmostEqual(
            self.loss(self.pred, self.gt, zero_mask).item(), 0)

        # Test 0 < fallback_negative_num < negative numbers
        all_neg_gt = torch.zeros((4, ))
        self.fallback_bce_loss = MaskedBalancedBCEWithLogitsLoss(
            fallback_negative_num=1)
        self.assertAlmostEqual(
            self.fallback_bce_loss(self.pred, all_neg_gt, self.mask).item(),
            -torch.log(1 - torch.sigmoid(logit)).item(),
            delta=0.001)
        # Test fallback_negative_num > negative numbers
        self.fallback_bce_loss = MaskedBalancedBCEWithLogitsLoss(
            fallback_negative_num=5)
        self.assertAlmostEqual(
            self.fallback_bce_loss(self.pred, all_neg_gt, self.mask).item(),
            -torch.log(1 - torch.sigmoid(logit)).item(),
            delta=0.001)


class TestMaskedBCEWithLogitsLoss(TestCase):

    def setUp(self) -> None:
        self.loss = MaskedBCEWithLogitsLoss()
        self.pred = torch.FloatTensor([1.5, 1.5, 1.5, 1.5])
        self.gt = torch.FloatTensor([1, 1, 1, 0])
        self.mask = torch.BoolTensor([True, False, False, True])

    def test_init(self):
        with self.assertRaises(AssertionError):
            MaskedBCEWithLogitsLoss(eps='a')

    def test_forward(self):

        # Shape mismatch between pred and gt
        with self.assertRaises(AssertionError):
            invalid_gt = torch.FloatTensor([0, 0, 0])
            self.loss(self.pred, invalid_gt)

        # Shape mismatch between pred and mask
        with self.assertRaises(AssertionError):
            invalid_mask = torch.BoolTensor([True, False, False])
            self.loss(self.pred, self.gt, invalid_mask)

        logit = torch.FloatTensor([1.5])
        self.assertAlmostEqual(
            self.loss(self.pred, self.gt).item(),
            ((-torch.log(torch.sigmoid(logit)) * 3 -
              torch.log(1 - torch.sigmoid(logit))) / 4).item(),
            delta=0.0001)
        self.assertAlmostEqual(
            self.loss(self.pred, self.gt, self.mask).item(),
            (-torch.log(torch.sigmoid(logit)) -
             torch.log(1 - torch.sigmoid(logit))).item() / 2,
            delta=0.0001)

        # Test zero mask
        zero_mask = torch.FloatTensor([0, 0, 0, 0])
        self.assertAlmostEqual(
            self.loss(self.pred, self.gt, zero_mask).item(), 0)

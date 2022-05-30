# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmocr.models.textdet.losses.common import MaskedDiceLoss


class TestMaskedDiceLoss(TestCase):

    def setUp(self) -> None:
        self.loss = MaskedDiceLoss()
        self.pred = torch.FloatTensor([0, 1, 0, 1])
        self.gt = torch.ones_like(self.pred)
        self.mask = torch.FloatTensor([1, 1, 0, 1])

    def test_init(self):
        with self.assertRaises(AssertionError):
            MaskedDiceLoss(eps='any')

    def test_forward(self):

        # Shape mismatch between pred and gt
        with self.assertRaises(AssertionError):
            invalid_gt = torch.FloatTensor([0, 0, 0])
            self.loss(self.pred, invalid_gt)

        # Shape mismatch between pred and mask
        with self.assertRaises(AssertionError):
            invalid_mask = torch.BoolTensor([True, False, False])
            self.loss(self.pred, self.gt, invalid_mask)

        self.assertAlmostEqual(
            self.loss(self.pred, self.gt).item(), 1 / 3, delta=0.001)
        self.assertAlmostEqual(
            self.loss(self.pred, self.gt, self.mask).item(),
            1 / 5,
            delta=0.001)

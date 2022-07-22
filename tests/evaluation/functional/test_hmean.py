# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from mmocr.evaluation.functional import compute_hmean


class TestHmean(TestCase):

    def test_compute_hmean(self):
        with self.assertRaises(AssertionError):
            compute_hmean(0, 0, 0.0, 0)
        with self.assertRaises(AssertionError):
            compute_hmean(0, 0, 0, 0.0)
        with self.assertRaises(AssertionError):
            compute_hmean([1], 0, 0, 0)
        with self.assertRaises(AssertionError):
            compute_hmean(0, [1], 0, 0)

        _, _, hmean = compute_hmean(2, 2, 2, 2)
        self.assertEqual(hmean, 1)

        _, _, hmean = compute_hmean(0, 0, 2, 2)
        self.assertEqual(hmean, 0)

# Copyright (c) OpenMMLab. All rights reserved.
import unittest

from mmocr.utils import track_parallel_progress_multi_args


def func(a, b):
    return a + b


class TestProcessing(unittest.TestCase):

    def test_track_parallel_progress_multi_args(self):

        args = ([1, 2, 3], [4, 5, 6])
        results = track_parallel_progress_multi_args(func, args, nproc=1)
        self.assertEqual(results, [5, 7, 9])

        results = track_parallel_progress_multi_args(func, args, nproc=2)
        self.assertEqual(results, [5, 7, 9])

        with self.assertRaises(AssertionError):
            track_parallel_progress_multi_args(func, 1, nproc=1)

        with self.assertRaises(AssertionError):
            track_parallel_progress_multi_args(func, ([1, 2], 1), nproc=1)

        with self.assertRaises(AssertionError):
            track_parallel_progress_multi_args(
                func, ([1, 2], [1, 2, 3]), nproc=1)

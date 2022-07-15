# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np

from mmocr.utils import crop_img, warp_img


class TestImgUtils(unittest.TestCase):

    def test_warp_img(self):
        dummy_img = np.ones((600, 600, 3), dtype=np.uint8)
        dummy_box = [20, 20, 120, 20, 120, 40, 20, 40]

        cropped_img = warp_img(dummy_img, dummy_box)

        with self.assertRaises(AssertionError):
            warp_img(dummy_img, [])
        with self.assertRaises(AssertionError):
            warp_img(dummy_img, [20, 40, 40, 20])

        self.assertAlmostEqual(cropped_img.shape[0], 20)
        self.assertAlmostEqual(cropped_img.shape[1], 100)

    def test_min_rect_crop(self):
        dummy_img = np.ones((600, 600, 3), dtype=np.uint8)
        dummy_box = [20, 20, 120, 20, 120, 40, 20, 40]

        cropped_img = crop_img(
            dummy_img,
            dummy_box,
            0.,
            0.,
        )

        with self.assertRaises(AssertionError):
            crop_img(dummy_img, [])
        with self.assertRaises(AssertionError):
            crop_img(dummy_img, [20, 40, 40, 20])
        with self.assertRaises(AssertionError):
            crop_img(dummy_img, dummy_box, 4, 0.2)
        with self.assertRaises(AssertionError):
            crop_img(dummy_img, dummy_box, 0.4, 1.2)

        self.assertAlmostEqual(cropped_img.shape[0], 20)
        self.assertAlmostEqual(cropped_img.shape[1], 100)

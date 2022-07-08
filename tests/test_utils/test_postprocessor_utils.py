# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np
import torch

from mmocr.utils import fill_hole


class TestFillHole(unittest.TestCase):

    def setUp(self) -> None:
        self.input_mask_list = [[0, 1, 1, 1, 0], [0, 1, 0, 1, 0],
                                [0, 1, 1, 1, 0]]
        self.input_mask_array = np.array(self.input_mask_list)
        self.input_mask_tensor = torch.tensor(self.input_mask_list)
        self.gt = np.array([[0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0]])

    def test_fill_hole(self):
        self.assertTrue(np.allclose(fill_hole(self.input_mask_list), self.gt))
        self.assertTrue(np.allclose(fill_hole(self.input_mask_array), self.gt))
        self.assertTrue(
            np.allclose(fill_hole(self.input_mask_tensor), self.gt))

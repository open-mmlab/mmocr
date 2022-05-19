# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np
import torch

from mmocr.utils import bbox2poly


class TestBboxUtils(unittest.TestCase):

    def test_bbox2poly(self):
        # test np.array
        box = np.array([0, 0, 1, 1])
        self.assertTrue(
            np.all(bbox2poly(box) == np.array([0, 0, 1, 0, 1, 1, 0, 1])))
        # test list
        box = [0, 0, 1, 1]
        self.assertTrue(
            np.all(bbox2poly(box) == np.array([0, 0, 1, 0, 1, 1, 0, 1])))
        # test tensor
        box = torch.Tensor([0, 0, 1, 1])
        self.assertTrue(
            np.all(bbox2poly(box) == np.array([0, 0, 1, 0, 1, 1, 0, 1])))

# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch
from parameterized import parameterized

from mmocr.models.textdet.necks import FPNF


class TestFPNF(unittest.TestCase):

    def setUp(self):
        in_channels = [256, 512, 1024, 2048]
        size = [112, 56, 28, 14]
        inputs = []
        for i in range(4):
            inputs.append(torch.rand(1, in_channels[i], size[i], size[i]))
        self.inputs = inputs

    @parameterized.expand([('concat'), ('add')])
    def test_forward(self, fusion_type):
        fpnf = FPNF(fusion_type=fusion_type)
        outputs = fpnf.forward(self.inputs)
        self.assertListEqual(list(outputs.size()), [1, 256, 112, 112])

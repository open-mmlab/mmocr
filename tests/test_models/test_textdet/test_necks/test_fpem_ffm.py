# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch

from mmocr.models.textdet.necks.fpem_ffm import FPEM, FPEM_FFM


class TestFPEM(unittest.TestCase):

    def setUp(self):
        self.c2 = torch.Tensor(1, 8, 64, 64)
        self.c3 = torch.Tensor(1, 8, 32, 32)
        self.c4 = torch.Tensor(1, 8, 16, 16)
        self.c5 = torch.Tensor(1, 8, 8, 8)
        self.fpem = FPEM(in_channels=8)

    def test_forward(self):
        neck = FPEM(in_channels=8)
        neck.init_weights()
        out = neck(self.c2, self.c3, self.c4, self.c5)
        self.assertTrue(out[0].shape == self.c2.shape)
        self.assertTrue(out[1].shape == self.c3.shape)
        self.assertTrue(out[2].shape == self.c4.shape)
        self.assertTrue(out[3].shape == self.c5.shape)


class TestFPEM_FFM(unittest.TestCase):

    def setUp(self):
        self.c2 = torch.Tensor(1, 8, 64, 64)
        self.c3 = torch.Tensor(1, 16, 32, 32)
        self.c4 = torch.Tensor(1, 32, 16, 16)
        self.c5 = torch.Tensor(1, 64, 8, 8)
        self.in_channels = [8, 16, 32, 64]
        self.conv_out = 8
        self.features = [self.c2, self.c3, self.c4, self.c5]

    def test_forward(self):
        neck = FPEM_FFM(in_channels=self.in_channels, conv_out=self.conv_out)
        neck.init_weights()
        out = neck(self.features)
        self.assertTrue(out[0].shape == torch.Size([1, 8, 64, 64]))
        self.assertTrue(out[1].shape == out[0].shape)
        self.assertTrue(out[2].shape == out[0].shape)
        self.assertTrue(out[3].shape == out[0].shape)


if __name__ == '__main__':
    model = TestFPEM_FFM()
    model.setUp()
    model.test_forward()

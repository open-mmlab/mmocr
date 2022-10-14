# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmocr.models.common.plugins import AvgPool2d


class TestAvgPool2d(TestCase):

    def setUp(self) -> None:
        self.img = torch.rand(1, 3, 32, 100)

    def test_avgpool2d(self):
        avgpool2d = AvgPool2d(kernel_size=2, stride=2)
        self.assertEqual(avgpool2d(self.img).shape, torch.Size([1, 3, 16, 50]))

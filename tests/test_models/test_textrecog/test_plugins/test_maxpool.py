# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmocr.models.textrecog.plugins import Maxpool2d


class TestMaxpool2d(TestCase):

    def setUp(self) -> None:
        self.img = torch.rand(1, 3, 32, 100)

    def test_maxpool2d(self):
        maxpool2d = Maxpool2d(kernel_size=2, stride=2)
        self.assertEqual(maxpool2d(self.img).shape, torch.Size([1, 3, 16, 50]))

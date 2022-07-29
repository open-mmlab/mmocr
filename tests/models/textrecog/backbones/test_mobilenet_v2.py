# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmocr.models.textrecog.backbones import MobileNetV2


class TestMobileNetV2(TestCase):

    def setUp(self) -> None:
        self.img = torch.rand(1, 3, 32, 160)

    def test_mobilenetv2(self):
        mobilenet_v2 = MobileNetV2()
        self.assertEqual(
            mobilenet_v2(self.img).shape, torch.Size([1, 1280, 1, 43]))

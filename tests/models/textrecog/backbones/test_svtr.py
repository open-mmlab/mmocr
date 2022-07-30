# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmocr.models.textrecog.backbones.svtr import OverlapPatchEmbed


class TestPatchEmbed(TestCase):

    def setUp(self) -> None:
        self.img = torch.rand(1, 3, 32, 100)

    def test_patch_embed(self):
        Patch_Embed = OverlapPatchEmbed(
            img_size=self.img.shape[-2:], in_channels=self.img.shape[1])
        self.assertEqual(
            Patch_Embed(self.img).shape, torch.Size([1, 8 * 25, 768]))

# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmocr.models.textrecog.backbones.svtr import (AttnMixer, ConvMixer,
                                                   OverlapPatchEmbed)


class TestOverlapPatchEmbed(TestCase):

    def setUp(self) -> None:
        self.img = torch.rand(1, 3, 32, 100)

    def test_overlap_patch_embed(self):
        Overlap_Patch_Embed = OverlapPatchEmbed(
            img_size=self.img.shape[-2:], in_channels=self.img.shape[1])
        self.assertEqual(
            Overlap_Patch_Embed(self.img).shape, torch.Size([1, 8 * 25, 768]))


class TestConvMixer(TestCase):

    def setUp(self) -> None:
        self.img = torch.rand(1, 8 * 25, 768)

    def test_conv_mixer(self):
        conv_mixer = ConvMixer(embed_dims=self.img.shape[-1])
        self.assertEqual(
            conv_mixer(self.img).shape, torch.Size([1, 8 * 25, 768]))


class TestAttnMixer(TestCase):

    def setUp(self) -> None:
        self.img = torch.rand(1, 8 * 25, 768)

    def test_attn_mixer(self):
        attn_mixer = AttnMixer(embed_dims=self.img.shape[-1])
        self.assertEqual(
            attn_mixer(self.img).shape, torch.Size([1, 8 * 25, 768]))

# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmocr.models.textrecog.backbones.svtr import (AttnMixer, ConvMixer,
                                                   MerigingBlock, MixingBlock,
                                                   OverlapPatchEmbed, SVTRNet)


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


class TestMixingBlock(TestCase):

    def setUp(self) -> None:
        self.img = torch.rand(1, 8 * 25, 768)

    def test_mixing_block(self):
        mixing_block = MixingBlock(self.img.shape[-1], num_heads=8)
        self.assertEqual(
            mixing_block(self.img).shape, torch.Size([1, 8 * 25, 768]))


class TestMergingBlock(TestCase):

    def setUp(self) -> None:
        self.img = torch.rand(1, 64, 8, 25)

    def test_mergingblock(self):
        mergingblock1 = MerigingBlock(
            self.img.shape[1], self.img.shape[1] * 2, types='Pool')
        mergingblock2 = MerigingBlock(
            self.img.shape[1], self.img.shape[1] * 2, types='Conv')
        self.assertEqual(
            [mergingblock1(self.img).shape,
             mergingblock2(self.img).shape],
            [torch.Size([1, 4 * 25, 64 * 2]),
             torch.Size([1, 4 * 25, 64 * 2])])


class TestSvtrNet(TestCase):

    def setUp(self) -> None:
        self.img = torch.rand(1, 3, 32, 100)

    def test_svtrnet(self):
        model = SVTRNet(
            img_size=self.img.shape[-2:],
            in_channels=self.img.shape[1],
        )
        model.train()
        self.assertEqual(model(self.img).shape, torch.Size([1, 192, 1, 25]))

# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmocr.models.common.layers.transformer_layers import (TFDecoderLayer,
                                                           TFEncoderLayer)


class TestTFEncoderLayer(TestCase):

    def test_forward(self):
        encoder_layer = TFEncoderLayer()
        in_enc = torch.rand(1, 20, 512)
        out_enc = encoder_layer(in_enc)
        self.assertEqual(out_enc.shape, torch.Size([1, 20, 512]))

        encoder_layer = TFEncoderLayer(
            operation_order=('self_attn', 'norm', 'ffn', 'norm'))
        out_enc = encoder_layer(in_enc)
        self.assertEqual(out_enc.shape, torch.Size([1, 20, 512]))


class TestTFDecoderLayer(TestCase):

    def test_forward(self):
        decoder_layer = TFDecoderLayer()
        in_dec = torch.rand(1, 30, 512)
        out_enc = torch.rand(1, 128, 512)
        out_dec = decoder_layer(in_dec, out_enc)
        self.assertEqual(out_dec.shape, torch.Size([1, 30, 512]))

        decoder_layer = TFDecoderLayer(
            operation_order=('self_attn', 'norm', 'enc_dec_attn', 'norm',
                             'ffn', 'norm'))
        out_dec = decoder_layer(in_dec, out_enc)
        self.assertEqual(out_dec.shape, torch.Size([1, 30, 512]))

# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmocr.structures import TextRecogDataSample
from mmocr.models.textrecog.encoders import SAREncoder


class TestSAREncoder(TestCase):

    def setUp(self):
        gt_text_sample1 = TextRecogDataSample()
        gt_text_sample1.set_metainfo(dict(valid_ratio=0.9))

        gt_text_sample2 = TextRecogDataSample()
        gt_text_sample2.set_metainfo(dict(valid_ratio=1.0))

        self.data_info = [gt_text_sample1, gt_text_sample2]

    def test_init(self):
        with self.assertRaises(AssertionError):
            SAREncoder(enc_bi_rnn='bi')
        with self.assertRaises(AssertionError):
            SAREncoder(rnn_dropout=2)
        with self.assertRaises(AssertionError):
            SAREncoder(enc_gru='gru')
        with self.assertRaises(AssertionError):
            SAREncoder(d_model=512.5)
        with self.assertRaises(AssertionError):
            SAREncoder(d_enc=200.5)
        with self.assertRaises(AssertionError):
            SAREncoder(mask='mask')

    def test_forward(self):
        encoder = SAREncoder()
        encoder.init_weights()
        encoder.train()

        feat = torch.randn(2, 512, 4, 40)
        with self.assertRaises(AssertionError):
            encoder(feat, self.data_info * 2)
        out_enc = encoder(feat, self.data_info)
        self.assertEqual(out_enc.shape, torch.Size([2, 512]))

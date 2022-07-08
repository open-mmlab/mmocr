# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.data import LabelData

from mmocr.core.data_structures import TextRecogDataSample
from mmocr.models.textrecog.decoders import (PositionAttentionDecoder,
                                             RobustScannerFuser,
                                             SequenceAttentionDecoder)


class TestRobustScannerFuser(TestCase):

    def setUp(self) -> None:
        gt_text_sample1 = TextRecogDataSample()
        gt_text = LabelData()
        gt_text.item = 'Hello'
        gt_text_sample1.gt_text = gt_text
        gt_text_sample1.set_metainfo(dict(valid_ratio=0.9))

        gt_text_sample2 = TextRecogDataSample()
        gt_text = LabelData()
        gt_text = LabelData()
        gt_text.item = 'World'
        gt_text_sample2.gt_text = gt_text
        gt_text_sample2.set_metainfo(dict(valid_ratio=1.0))

        self.data_info = [gt_text_sample1, gt_text_sample2]
        self.dict_cfg = dict(
            type='Dictionary',
            dict_file='dicts/lower_english_digits.txt',
            with_start=True,
            with_end=True,
            same_start_end=True,
            with_padding=True,
            with_unknown=True)

        self.loss_cfg = dict(type='CELoss')
        hybrid_decoder = dict(type='SequenceAttentionDecoder')
        position_decoder = dict(type='PositionAttentionDecoder')
        self.decoder = RobustScannerFuser(
            dictionary=self.dict_cfg,
            loss_module=self.loss_cfg,
            hybrid_decoder=hybrid_decoder,
            position_decoder=position_decoder,
            max_seq_len=40)

    def test_init(self):

        self.assertIsInstance(self.decoder.hybrid_decoder,
                              SequenceAttentionDecoder)
        self.assertIsInstance(self.decoder.position_decoder,
                              PositionAttentionDecoder)
        hybrid_decoder = dict(type='SequenceAttentionDecoder', max_seq_len=40)
        position_decoder = dict(type='PositionAttentionDecoder')
        with self.assertWarns(Warning):
            RobustScannerFuser(
                dictionary=self.dict_cfg,
                loss_module=self.loss_cfg,
                hybrid_decoder=hybrid_decoder,
                position_decoder=position_decoder,
                max_seq_len=40)
        hybrid_decoder = dict(
            type='SequenceAttentionDecoder', dictionary=self.dict_cfg)
        with self.assertWarns(Warning):
            RobustScannerFuser(
                dictionary=self.dict_cfg,
                loss_module=self.loss_cfg,
                hybrid_decoder=hybrid_decoder,
                position_decoder=position_decoder,
                max_seq_len=40)

    def test_forward_train(self):
        feat = torch.randn(2, 512, 8, 8)
        encoder_out = torch.randn(2, 128, 8, 8)
        self.decoder.train()
        output = self.decoder(
            feat=feat, out_enc=encoder_out, data_samples=self.data_info)
        self.assertTupleEqual(tuple(output.shape), (2, 40, 39))

    def test_forward_test(self):
        feat = torch.randn(2, 512, 8, 8)
        encoder_out = torch.randn(2, 128, 8, 8)
        self.decoder.eval()
        output = self.decoder(
            feat=feat, out_enc=encoder_out, data_samples=self.data_info)
        self.assertTupleEqual(tuple(output.shape), (2, 40, 39))

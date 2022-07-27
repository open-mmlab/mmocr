# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.data import LabelData

from mmocr.data import TextRecogDataSample
from mmocr.models.textrecog.decoders import SequenceAttentionDecoder


class TestSequenceAttentionDecoder(TestCase):

    def setUp(self):
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

    def test_init(self):

        module_loss_cfg = dict(type='CEModuleLoss')
        decoder = SequenceAttentionDecoder(
            dictionary=self.dict_cfg,
            module_loss=module_loss_cfg,
            return_feature=False)
        self.assertIsInstance(decoder.prediction, torch.nn.Linear)

    def test_forward_train(self):
        feat = torch.randn(2, 512, 8, 8)
        encoder_out = torch.randn(2, 128, 8, 8)
        module_loss_cfg = dict(type='CEModuleLoss')
        decoder = SequenceAttentionDecoder(
            dictionary=self.dict_cfg,
            module_loss=module_loss_cfg,
            return_feature=False)
        data_samples = decoder.module_loss.get_targets(self.data_info)
        output = decoder.forward_train(
            feat=feat, out_enc=encoder_out, data_samples=data_samples)
        self.assertTupleEqual(tuple(output.shape), (2, 40, 39))

        decoder = SequenceAttentionDecoder(
            dictionary=self.dict_cfg, module_loss=module_loss_cfg)
        output = decoder.forward_train(
            feat=feat, out_enc=encoder_out, data_samples=data_samples)
        self.assertTupleEqual(tuple(output.shape), (2, 40, 512))

        feat_new = torch.randn(2, 256, 8, 8)
        with self.assertRaises(AssertionError):
            decoder.forward_train(feat_new, encoder_out, self.data_info)
        encoder_out_new = torch.randn(2, 256, 8, 8)
        with self.assertRaises(AssertionError):
            decoder.forward_train(feat, encoder_out_new, self.data_info)

    def test_forward_test(self):
        feat = torch.randn(2, 512, 8, 8)
        encoder_out = torch.randn(2, 128, 8, 8)
        module_loss_cfg = dict(type='CEModuleLoss')
        decoder = SequenceAttentionDecoder(
            dictionary=self.dict_cfg,
            module_loss=module_loss_cfg,
            return_feature=False)
        output = decoder.forward_test(feat, encoder_out, self.data_info)
        self.assertTupleEqual(tuple(output.shape), (2, 40, 39))

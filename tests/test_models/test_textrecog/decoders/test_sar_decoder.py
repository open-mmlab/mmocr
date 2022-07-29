# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.data import LabelData

from mmocr.data import TextRecogDataSample
from mmocr.models.textrecog.decoders import (ParallelSARDecoder,
                                             SequentialSARDecoder)


class TestParallelSARDecoder(TestCase):

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
        self.max_seq_len = 40

    def test_init(self):
        decoder = ParallelSARDecoder(self.dict_cfg)
        self.assertIsInstance(decoder.rnn_decoder, torch.nn.LSTM)
        decoder = ParallelSARDecoder(
            self.dict_cfg, dec_gru=True, pred_concat=True)
        self.assertIsInstance(decoder.rnn_decoder, torch.nn.GRU)

    def test_forward_train(self):
        # test parallel sar decoder
        loss_cfg = dict(type='CEModuleLoss')
        decoder = ParallelSARDecoder(
            self.dict_cfg, module_loss=loss_cfg, max_seq_len=self.max_seq_len)
        decoder.init_weights()
        decoder.train()
        feat = torch.rand(2, 512, 4, self.max_seq_len)
        out_enc = torch.rand(2, 512)
        data_samples = decoder.module_loss.get_targets(self.data_info)
        decoder.train_mode = True
        out_train = decoder.forward_train(
            feat, out_enc, data_samples=data_samples)
        self.assertEqual(out_train.shape, torch.Size([2, self.max_seq_len,
                                                      39]))

    def test_forward_test(self):
        decoder = ParallelSARDecoder(
            self.dict_cfg, max_seq_len=self.max_seq_len)
        feat = torch.rand(2, 512, 4, self.max_seq_len)
        out_enc = torch.rand(2, 512)
        decoder.train_mode = False
        out_test = decoder.forward_test(feat, out_enc, self.data_info)
        assert out_test.shape == torch.Size([2, self.max_seq_len, 39])
        out_test = decoder.forward_test(feat, out_enc, None)
        assert out_test.shape == torch.Size([2, self.max_seq_len, 39])


class TestSequentialSARDecoder(TestCase):

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
        decoder = SequentialSARDecoder(self.dict_cfg)
        self.assertIsInstance(decoder.rnn_decoder_layer1, torch.nn.LSTMCell)
        decoder = SequentialSARDecoder(
            self.dict_cfg, dec_gru=True, pred_concat=True)
        self.assertIsInstance(decoder.rnn_decoder_layer1, torch.nn.GRUCell)

    def test_forward_train(self):
        # test parallel sar decoder
        loss_cfg = dict(type='CEModuleLoss')
        decoder = SequentialSARDecoder(self.dict_cfg, module_loss=loss_cfg)
        decoder.init_weights()
        decoder.train()
        feat = torch.rand(2, 512, 4, 40)
        out_enc = torch.rand(2, 512)
        data_samples = decoder.module_loss.get_targets(self.data_info)
        out_train = decoder.forward_train(feat, out_enc, data_samples)
        self.assertEqual(out_train.shape, torch.Size([2, 40, 39]))

    def test_forward_test(self):
        # test parallel sar decoder
        loss_cfg = dict(type='CEModuleLoss')
        decoder = SequentialSARDecoder(self.dict_cfg, module_loss=loss_cfg)
        decoder.init_weights()
        decoder.train()
        feat = torch.rand(2, 512, 4, 40)
        out_enc = torch.rand(2, 512)
        out_test = decoder.forward_test(feat, out_enc, self.data_info)
        self.assertEqual(out_test.shape, torch.Size([2, 40, 39]))
        out_test = decoder.forward_test(feat, out_enc, None)
        self.assertEqual(out_test.shape, torch.Size([2, 40, 39]))

# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from unittest import TestCase

import torch
from mmengine.data import LabelData

from mmocr.structures import TextRecogDataSample
from mmocr.models.textrecog.decoders import NRTRDecoder
from mmocr.testing import create_dummy_dict_file


class TestNRTRDecoder(TestCase):

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

    def test_init(self):
        tmp_dir = tempfile.TemporaryDirectory()
        dict_file = osp.join(tmp_dir.name, 'fake_chars.txt')
        create_dummy_dict_file(dict_file)
        dict_cfg = dict(
            type='Dictionary',
            dict_file=dict_file,
            with_start=True,
            with_end=True,
            same_start_end=True,
            with_padding=True,
            with_unknown=True)
        loss_cfg = dict(type='CEModuleLoss')
        NRTRDecoder(dictionary=dict_cfg, module_loss=loss_cfg)
        tmp_dir.cleanup()

    def test_forward_train(self):
        encoder_out = torch.randn(2, 25, 512)
        tmp_dir = tempfile.TemporaryDirectory()
        max_seq_len = 40
        dict_file = osp.join(tmp_dir.name, 'fake_chars.txt')
        create_dummy_dict_file(dict_file)
        # test diction cfg
        dict_cfg = dict(
            type='Dictionary',
            dict_file=dict_file,
            with_start=True,
            with_end=True,
            same_start_end=True,
            with_padding=True,
            with_unknown=True)
        loss_cfg = dict(type='CEModuleLoss')
        decoder = NRTRDecoder(
            dictionary=dict_cfg, module_loss=loss_cfg, max_seq_len=max_seq_len)
        data_samples = decoder.module_loss.get_targets(self.data_info)
        output = decoder.forward_train(
            out_enc=encoder_out, data_samples=data_samples)
        self.assertTupleEqual(tuple(output.shape), (2, max_seq_len, 39))

    def test_forward_test(self):
        encoder_out = torch.randn(2, 25, 512)
        tmp_dir = tempfile.TemporaryDirectory()
        dict_file = osp.join(tmp_dir.name, 'fake_chars.txt')
        create_dummy_dict_file(dict_file)
        # test diction cfg
        dict_cfg = dict(
            type='Dictionary',
            dict_file=dict_file,
            with_start=True,
            with_end=True,
            same_start_end=True,
            with_padding=True,
            with_unknown=True)
        loss_cfg = dict(type='CEModuleLoss')
        decoder = NRTRDecoder(
            dictionary=dict_cfg, module_loss=loss_cfg, max_seq_len=40)
        output = decoder.forward_test(
            out_enc=encoder_out, data_samples=self.data_info)
        self.assertTupleEqual(tuple(output.shape), (2, 40, 39))

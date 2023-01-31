# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from unittest import TestCase

import torch

from mmengine.structures import LabelData
from mmocr.models.textrecog.decoders.svtr_decoder import SVTRDecoder
from mmocr.structures import TextRecogDataSample
from mmocr.testing import create_dummy_dict_file


class TestSVTRDecoder(TestCase):

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
        with tempfile.TemporaryDirectory() as tmp_dir:
            dict_file = osp.join(tmp_dir, 'fake_chars.txt')
            create_dummy_dict_file(dict_file)
            dict_cfg = dict(
                type='Dictionary',
                dict_file=dict_file,
                with_start=True,
                with_end=True,
                same_start_end=True,
                with_padding=True,
                with_unknown=True)
            loss_cfg = dict(type='CTCModuleLoss', letter_case='lower')
            SVTRDecoder(
                in_channels=192, dictionary=dict_cfg, module_loss=loss_cfg)

    def test_forward_train(self):
        out_enc = torch.randn(1, 192, 1, 25)
        tmp_dir = tempfile.TemporaryDirectory()
        max_seq_len = 25
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
        loss_cfg = dict(type='CTCModuleLoss', letter_case='lower')
        decoder = SVTRDecoder(
            in_channels=192,
            dictionary=dict_cfg,
            module_loss=loss_cfg,
            max_seq_len=max_seq_len,
        )
        data_samples = decoder.module_loss.get_targets(self.data_info)
        output = decoder.forward_train(
            out_enc=out_enc, data_samples=data_samples)
        self.assertTupleEqual(tuple(output.shape), (1, max_seq_len, 39))

    def test_forward_test(self):
        out_enc = torch.randn(1, 192, 1, 25)
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
        loss_cfg = dict(type='CTCModuleLoss', letter_case='lower')
        decoder = SVTRDecoder(
            in_channels=192,
            dictionary=dict_cfg,
            module_loss=loss_cfg,
            max_seq_len=25)
        output = decoder.forward_test(
            out_enc=out_enc, data_samples=self.data_info)
        self.assertTupleEqual(tuple(output.shape), (1, 25, 39))

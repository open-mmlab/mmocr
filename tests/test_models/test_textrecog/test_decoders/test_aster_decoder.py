# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from unittest import TestCase

import torch
from mmengine.structures import LabelData

from mmocr.models.textrecog.decoders import ASTERDecoder
from mmocr.structures import TextRecogDataSample


class TestASTERDecoder(TestCase):

    def setUp(self):
        gt_text_sample1 = TextRecogDataSample()
        gt_text = LabelData()
        gt_text.item = 'Hello'
        gt_text_sample1.gt_text = gt_text

        gt_text_sample2 = TextRecogDataSample()
        gt_text = LabelData()
        gt_text = LabelData()
        gt_text.item = 'World1'
        gt_text_sample2.gt_text = gt_text

        self.data_info = [gt_text_sample1, gt_text_sample2]

    def _create_dummy_dict_file(
        self, dict_file,
        chars=list('0123456789abcdefghijklmnopqrstuvwxyz')):  # NOQA
        with open(dict_file, 'w') as f:
            for char in chars:
                f.write(char + '\n')

    def test_init(self):
        tmp_dir = tempfile.TemporaryDirectory()
        dict_file = osp.join(tmp_dir.name, 'fake_chars.txt')
        self._create_dummy_dict_file(dict_file)
        dict_cfg = dict(
            type='Dictionary',
            dict_file=dict_file,
            with_start=True,
            with_end=True,
            same_start_end=True,
            with_padding=True,
            with_unknown=True)
        loss_cfg = dict(type='CEModuleLoss')
        ASTERDecoder(
            in_channels=512, dictionary=dict_cfg, module_loss=loss_cfg)
        tmp_dir.cleanup()

    def test_forward_train(self):
        encoder_out = torch.randn(2, 25, 512)
        tmp_dir = tempfile.TemporaryDirectory()
        dict_file = osp.join(tmp_dir.name, 'fake_chars.txt')
        self._create_dummy_dict_file(dict_file)
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
        decoder = ASTERDecoder(
            in_channels=512,
            dictionary=dict_cfg,
            module_loss=loss_cfg,
            max_seq_len=25)
        data_samples = decoder.module_loss.get_targets(self.data_info)
        output = decoder.forward_train(
            out_enc=encoder_out, data_samples=data_samples)
        self.assertTupleEqual(tuple(output.shape), (2, 25, 39))

    def test_forward_test(self):
        encoder_out = torch.randn(2, 25, 512)
        tmp_dir = tempfile.TemporaryDirectory()
        dict_file = osp.join(tmp_dir.name, 'fake_chars.txt')
        self._create_dummy_dict_file(dict_file)
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
        decoder = ASTERDecoder(
            in_channels=512,
            dictionary=dict_cfg,
            module_loss=loss_cfg,
            max_seq_len=25)
        output = decoder.forward_test(
            out_enc=encoder_out, data_samples=self.data_info)
        self.assertTupleEqual(tuple(output.shape), (2, 25, 39))

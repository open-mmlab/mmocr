# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from unittest import TestCase

import torch
import torch.nn as nn

from mmocr.models.textrecog.decoders import CRNNDecoder


class TestCRNNDecoder(TestCase):

    def _create_dummy_dict_file(
        self, dict_file,
        chars=list('0123456789abcdefghijklmnopqrstuvwxyz')):  # NOQA
        with open(dict_file, 'w') as f:
            for char in chars:
                f.write(char + '\n')

    def test_init(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dict_file = osp.join(tmp_dir, 'fake_chars.txt')
            self._create_dummy_dict_file(dict_file)
            # test diction cfg
            dict_cfg = dict(
                type='Dictionary',
                dict_file=dict_file,
                with_start=True,
                with_end=True,
                same_start_end=False,
                with_padding=True,
                with_unknown=True)
            # test rnn flag false
            decoder = CRNNDecoder(in_channels=3, dictionary=dict_cfg)
            self.assertIsInstance(decoder.decoder, nn.Conv2d)

            decoder = CRNNDecoder(
                in_channels=3, dictionary=dict_cfg, rnn_flag=True)
            self.assertIsInstance(decoder.decoder, nn.Sequential)

    def test_forward(self):
        inputs = torch.randn(3, 10, 1, 100)
        with tempfile.TemporaryDirectory() as tmp_dir:
            dict_file = osp.join(tmp_dir, 'fake_chars.txt')
            self._create_dummy_dict_file(dict_file)
            # test diction cfg
            dict_cfg = dict(
                type='Dictionary',
                dict_file=dict_file,
                with_start=False,
                with_end=False,
                same_start_end=False,
                with_padding=True,
                with_unknown=False)
            decoder = CRNNDecoder(in_channels=10, dictionary=dict_cfg)
            output = decoder(inputs, train_mode=True)
            self.assertTupleEqual(tuple(output.shape), (3, 100, 37))
            decoder = CRNNDecoder(
                in_channels=10, dictionary=dict_cfg, rnn_flag=True)
            output = decoder(inputs, train_mode=False)
            self.assertTupleEqual(tuple(output.shape), (3, 100, 37))

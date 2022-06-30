# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from unittest import TestCase

import numpy as np
import torch
from mmengine.data import LabelData

from mmocr.core.data_structures import TextRecogDataSample
from mmocr.models.textrecog.dictionary import Dictionary
from mmocr.models.textrecog.losses import BaseRecogLoss


class TestBaseRecogLoss(TestCase):

    def _create_dummy_dict_file(
        self, dict_file,
        chars=list('0123456789abcdefghijklmnopqrstuvwxyz')):  # NOQA
        with open(dict_file, 'w') as f:
            for char in chars:
                f.write(char + '\n')

    def _equal(self, a, b):
        if isinstance(a, (torch.Tensor, np.ndarray)):
            return (a == b).all()
        else:
            return a == b

    def test_init(self):
        tmp_dir = tempfile.TemporaryDirectory()
        dict_file = osp.join(tmp_dir.name, 'fake_chars.txt')
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
        base_recog_loss = BaseRecogLoss(dict_cfg)
        self.assertIsInstance(base_recog_loss.dictionary, Dictionary)
        # test case mode
        with self.assertRaises(AssertionError):
            base_recog_loss = BaseRecogLoss(dict_cfg, letter_case='no')
        # test dictionary is invalid type
        dict_cfg = ['tmp']
        with self.assertRaisesRegex(
                TypeError, ('The type of dictionary should be `Dictionary`'
                            ' or dict, '
                            f'but got {type(dict_cfg)}')):
            base_recog_loss = BaseRecogLoss(dict_cfg)

        tmp_dir.cleanup()

    def test_get_targets(self):
        label_data = LabelData(item='0123')
        data_sample = TextRecogDataSample()
        data_sample.gt_text = label_data
        tmp_dir = tempfile.TemporaryDirectory()
        dict_file = osp.join(tmp_dir.name, 'fake_chars.txt')
        self._create_dummy_dict_file(dict_file)
        # test diction cfg
        dictionary = Dictionary(
            dict_file=dict_file,
            with_start=True,
            with_end=True,
            same_start_end=False,
            with_padding=True,
            with_unknown=True)
        base_recog_loss = BaseRecogLoss(dictionary, max_seq_len=10)
        target_data_samples = base_recog_loss.get_targets([data_sample])
        assert self._equal(target_data_samples[0].gt_text.indexes,
                           torch.LongTensor([0, 1, 2, 3]))
        padding_idx = dictionary.padding_idx
        assert self._equal(
            target_data_samples[0].gt_text.padded_indexes,
            torch.LongTensor([
                dictionary.start_idx, 0, 1, 2, 3, dictionary.end_idx,
                padding_idx, padding_idx, padding_idx, padding_idx
            ]))
        self.assertTrue(target_data_samples[0].have_target)
        target_data_samples = base_recog_loss.get_targets(target_data_samples)
        data_sample.set_metainfo(dict(have_target=False))
        dictionary = Dictionary(
            dict_file=dict_file,
            with_start=False,
            with_end=False,
            same_start_end=False,
            with_padding=True,
            with_unknown=True)
        base_recog_loss = BaseRecogLoss(dictionary, max_seq_len=3)
        data_sample.gt_text.item = '0123'
        target_data_samples = base_recog_loss.get_targets([data_sample])
        assert self._equal(target_data_samples[0].gt_text.indexes,
                           torch.LongTensor([0, 1, 2, 3]))
        padding_idx = dictionary.padding_idx
        assert self._equal(target_data_samples[0].gt_text.padded_indexes,
                           torch.LongTensor([0, 1, 2]))
        data_sample.set_metainfo(dict(have_target=False))
        dict_cfg = dict(
            type='Dictionary',
            dict_file=dict_file,
            with_start=False,
            with_end=False,
            same_start_end=False,
            with_padding=False,
            with_unknown=True)
        base_recog_loss = BaseRecogLoss(
            dict_cfg, max_seq_len=10, letter_case='lower')
        data_sample.gt_text.item = '0123'
        target_data_samples = base_recog_loss.get_targets([data_sample])
        assert self._equal(target_data_samples[0].gt_text.indexes,
                           torch.LongTensor([0, 1, 2, 3]))
        assert self._equal(target_data_samples[0].gt_text.padded_indexes,
                           torch.LongTensor([0, 1, 2, 3]))

        target_data_samples = base_recog_loss.get_targets([])
        self.assertListEqual(target_data_samples, [])
        tmp_dir.cleanup()

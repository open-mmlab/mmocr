# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from unittest import TestCase

import numpy as np
import torch
from mmengine.data import LabelData

from mmocr.data import TextRecogDataSample
from mmocr.models.textrecog.dictionary import Dictionary
from mmocr.models.textrecog.losses import BaseRecogLoss
from mmocr.testing import create_dummy_dict_file


class TestBaseRecogLoss(TestCase):

    def _equal(self, a, b):
        if isinstance(a, (torch.Tensor, np.ndarray)):
            return (a == b).all()
        else:
            return a == b

    def test_init(self):
        tmp_dir = tempfile.TemporaryDirectory()
        dict_file = osp.join(tmp_dir.name, 'fake_chars.txt')
        create_dummy_dict_file(dict_file)
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
        # test invalid pad_with
        with self.assertRaises(AssertionError):
            base_recog_loss = BaseRecogLoss(dict_cfg, pad_with='test')
        # test invalid combination of dictionary and pad_with
        dict_cfg = dict(type='Dictionary', dict_file=dict_file, with_end=False)
        for pad_with in ['end', 'padding']:
            with self.assertRaisesRegex(
                    ValueError, f'pad_with="{pad_with}", but'
                    f' dictionary.{pad_with}_idx is None'):
                base_recog_loss = BaseRecogLoss(dict_cfg, pad_with=pad_with)
        with self.assertRaisesRegex(
                ValueError, 'pad_with="auto", but'
                ' dictionary.end_idx and dictionary.padding_idx are both'
                ' None'):
            base_recog_loss = BaseRecogLoss(dict_cfg, pad_with='auto')

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
        create_dummy_dict_file(dict_file)
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
            dict_file='dicts/lower_english_digits.txt',
            with_start=False,
            with_end=True,
            same_start_end=False,
            with_padding=True,
            with_unknown=True)
        base_recog_loss = BaseRecogLoss(
            dict_cfg, max_seq_len=10, letter_case='lower', pad_with='none')
        data_sample.gt_text.item = '0123'
        target_data_samples = base_recog_loss.get_targets([data_sample])
        assert self._equal(target_data_samples[0].gt_text.indexes,
                           torch.LongTensor([0, 1, 2, 3]))
        assert self._equal(target_data_samples[0].gt_text.padded_indexes,
                           torch.LongTensor([0, 1, 2, 3, 36]))

        target_data_samples = base_recog_loss.get_targets([])
        self.assertListEqual(target_data_samples, [])

        tmp_dir.cleanup()

# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from unittest import TestCase

from mmocr.models.textrecog import Dictionary


class TestDictionary(TestCase):

    def _create_dummy_dict_file(
        self, dict_file,
        chars=list('0123456789abcdefghijklmnopqrstuvwxyz')):  # NOQA
        with open(dict_file, 'w') as f:
            for char in chars:
                f.write(char + '\n')

    def test_init(self):
        tmp_dir = tempfile.TemporaryDirectory()

        # create dummy data
        dict_file = osp.join(tmp_dir.name, 'fake_chars.txt')
        self._create_dummy_dict_file(dict_file)
        # with start
        dict_gen = Dictionary(
            dict_file=dict_file,
            with_start=True,
            with_end=True,
            same_start_end=False,
            with_padding=True,
            with_unknown=True)
        self.assertEqual(dict_gen.num_classes, 40)
        self.assertListEqual(
            dict_gen.dict,
            list('0123456789abcdefghijklmnopqrstuvwxyz') +
            ['<BOS>', '<EOS>', '<PAD>', '<UKN>'])
        dict_gen = Dictionary(
            dict_file=dict_file,
            with_start=True,
            with_end=True,
            same_start_end=True,
            with_padding=True,
            with_unknown=True)
        assert dict_gen.num_classes == 39
        assert dict_gen.dict == list('0123456789abcdefghijklmnopqrstuvwxyz'
                                     ) + ['<BOS/EOS>', '<PAD>', '<UKN>']
        self.assertEqual(dict_gen.num_classes, 39)
        self.assertListEqual(
            dict_gen.dict,
            list('0123456789abcdefghijklmnopqrstuvwxyz') +
            ['<BOS/EOS>', '<PAD>', '<UKN>'])
        dict_gen = Dictionary(
            dict_file=dict_file,
            with_start=True,
            with_end=True,
            same_start_end=False,
            with_padding=True,
            with_unknown=True,
            start_token='<STA>',
            end_token='<END>',
            padding_token='<BLK>',
            unknown_token='<NO>')
        assert dict_gen.num_classes == 40
        assert dict_gen.dict[-4:] == ['<STA>', '<END>', '<BLK>', '<NO>']
        self.assertEqual(dict_gen.num_classes, 40)
        self.assertListEqual(dict_gen.dict[-4:],
                             ['<STA>', '<END>', '<BLK>', '<NO>'])
        dict_gen = Dictionary(
            dict_file=dict_file,
            with_start=True,
            with_end=True,
            same_start_end=True,
            with_padding=True,
            with_unknown=True,
            start_end_token='<BE>')
        self.assertEqual(dict_gen.num_classes, 39)
        self.assertListEqual(dict_gen.dict[-3:], ['<BE>', '<PAD>', '<UKN>'])
        # test len(line) > 1
        self._create_dummy_dict_file(dict_file, chars=['12', '3', '4'])
        with self.assertRaises(ValueError):
            dict_gen = Dictionary(dict_file=dict_file)

        # test duplicated dict
        self._create_dummy_dict_file(dict_file, chars=['1', '1', '2'])
        with self.assertRaises(AssertionError):
            dict_gen = Dictionary(dict_file=dict_file)

        tmp_dir.cleanup()

    def test_num_classes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # create dummy data
            dict_file = osp.join(tmp_dir, 'fake_chars.txt')
            self._create_dummy_dict_file(dict_file)
            dict_gen = Dictionary(dict_file=dict_file)
            assert dict_gen.num_classes == 36

    def test_contain_uppercase(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # create dummy data
            dict_file = osp.join(tmp_dir, 'fake_chars.txt')
            self._create_dummy_dict_file(dict_file)
            dict_gen = Dictionary(dict_file=dict_file)
            assert dict_gen.contain_uppercase is False
            self._create_dummy_dict_file(dict_file, chars='abcdABCD')
            dict_gen = Dictionary(dict_file=dict_file)
            assert dict_gen.contain_uppercase is True

    def test_str2idx(self):
        with tempfile.TemporaryDirectory() as tmp_dir:

            # create dummy data
            dict_file = osp.join(tmp_dir, 'fake_chars.txt')
            self._create_dummy_dict_file(dict_file)
            dict_gen = Dictionary(dict_file=dict_file)
            self.assertEqual(dict_gen.str2idx('01234'), [0, 1, 2, 3, 4])
            with self.assertRaises(Exception):
                dict_gen.str2idx('H')

            dict_gen = Dictionary(dict_file=dict_file, with_unknown=True)
            self.assertListEqual(dict_gen.str2idx('H'), [dict_gen.unknown_idx])

            dict_gen = Dictionary(
                dict_file=dict_file, with_unknown=True, unknown_token=None)
            self.assertListEqual(dict_gen.str2idx('H'), [])

    def test_idx2str(self):
        with tempfile.TemporaryDirectory() as tmp_dir:

            # create dummy data
            dict_file = osp.join(tmp_dir, 'fake_chars.txt')
            self._create_dummy_dict_file(dict_file)
            dict_gen = Dictionary(dict_file=dict_file)
            self.assertEqual(dict_gen.idx2str([0, 1, 2, 3, 4]), '01234')
            with self.assertRaises(AssertionError):
                dict_gen.idx2str('01234')

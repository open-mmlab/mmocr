# Copyright (c) OpenMMLab. All rights reserved.
import os
import tempfile
from unittest import TestCase

import lmdb

from mmocr.datasets import RecogLMDBDataset


class TestRecogLMDBDataset(TestCase):

    def create_deprecated_format_lmdb(self, temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
        env = lmdb.open(temp_dir, map_size=102400)
        cache = [(str(0).encode('utf-8'), b'test test')]
        with env.begin(write=True) as txn:
            cursor = txn.cursor()
            cursor.putmulti(cache, dupdata=False, overwrite=True)

        cache = []
        cache.append((b'total_number', str(1).encode('utf-8')))
        with env.begin(write=True) as txn:
            cursor = txn.cursor()
            cursor.putmulti(cache, dupdata=False, overwrite=True)

    def test_label_only_dataset(self):

        # test initialization
        dataset = RecogLMDBDataset(
            ann_file='tests/data/rec_toy_dataset/label.lmdb',
            data_prefix=dict(img_path='imgs'),
            pipeline=[])
        dataset.full_init()
        self.assertEqual(len(dataset), 10)
        self.assertEqual(len(dataset.load_data_list()), 10)

        # test load_data_list
        anno = dataset.load_data_list()[0]
        self.assertIn(anno['img_path'],
                      ['imgs/1223731.jpg', 'imgs\\1223731.jpg'])
        self.assertEqual(anno['instances'][0]['text'], 'GRAND')

    def test_label_and_image_dataset(self):

        # test initialization
        dataset = RecogLMDBDataset(
            ann_file='tests/data/rec_toy_dataset/imgs.lmdb',
            data_prefix=dict(img_path='imgs'),
            pipeline=[])
        dataset.full_init()
        self.assertEqual(len(dataset), 10)
        self.assertEqual(len(dataset.load_data_list()), 10)

        # test load_data_list
        anno = dataset.load_data_list()[0]
        self.assertIn(anno['img_path'],
                      [f'imgs/image-{1:09d}', f'imgs\\image-{1:09d}'])
        self.assertEqual(anno['instances'][0]['text'], 'GRAND')

    def test_deprecated_format(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.create_deprecated_format_lmdb(
                os.path.join(tmpdirname, 'data'))
            dataset = RecogLMDBDataset(
                ann_file=os.path.join(tmpdirname, 'data'),
                data_prefix=dict(img_path='imgs'),
                pipeline=[])

            warm_msg = 'DeprecationWarning: The lmdb dataset generated with '
            warm_msg += 'txt2lmdb will be deprecate, please use the latest '
            warm_msg += 'tools/data/utils/recog2lmdb to generate lmdb dataset.'
            warm_msg += ' See https://mmocr.readthedocs.io/en/'
            warm_msg += 'latest/tools.html#'
            warm_msg += 'convert-text-recognition-dataset-to-lmdb-format for '
            warm_msg += 'details.'

            dataset.full_init()
            self.assertWarnsRegex(UserWarning, warm_msg)
            dataset.close()

# Copyright (c) OpenMMLab. All rights reserved.
import os
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
        self.assertIn(anno['img_path'], [
            f'tests/data/rec_toy_dataset/imgs.lmdb/image-{1:09d}',
            f'tests\\data\\rec_toy_dataset\\imgs.lmdb\\image-{1:09d}'
        ])
        self.assertEqual(anno['instances'][0]['text'], 'GRAND')

# Copyright (c) OpenMMLab. All rights reserved.
import os
import tempfile
from unittest import TestCase

import cv2
import lmdb
import numpy as np

from mmocr.datasets import RecogLMDBDataset


class TestRecogLMDBDataset(TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        tmp_image = np.zeros((100, 100, 3), dtype=np.uint8)
        img_path = os.path.join(self.temp_dir.name, 'image.jpg')
        cv2.imwrite(img_path, tmp_image)
        with open(img_path, 'rb') as f:
            img_bin = f.read()
        self.lmdb_path = os.path.join(self.temp_dir.name, 'lmdb')
        env = lmdb.open(self.lmdb_path, map_size=102400)
        self.total_number = 5

        with env.begin(write=True) as txn:
            for idx in range(self.total_number):
                txn.put(f'label-{idx + 1:09d}'.encode('utf-8'), b'mmocr')
                txn.put(f'image-{idx + 1:09d}'.encode('utf-8'), img_bin)
            txn.put('num-samples'.encode('utf-8'),
                    str(self.total_number).encode('utf-8'))
        env.close()

    def test_label_and_image_dataset(self):

        # test initialization
        dataset = RecogLMDBDataset(ann_file=self.lmdb_path, pipeline=[])
        dataset.full_init()
        self.assertEqual(len(dataset), self.total_number)
        self.assertEqual(len(dataset.load_data_list()), self.total_number)

        for data in dataset:
            self.assertEqual(data['img'].shape, (100, 100, 3))
            self.assertEqual(data['instances'][0]['text'], 'mmocr')

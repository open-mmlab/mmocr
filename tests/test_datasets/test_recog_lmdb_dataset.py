# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from mmocr.datasets import RecogLMDBDataset


class TestRecogLMDBDataset(TestCase):

    def test_label_and_image_dataset(self):

        # test initialization
        dataset = RecogLMDBDataset(
            ann_file='tests/data/rec_toy_dataset/imgs.lmdb', pipeline=[])
        dataset.full_init()
        self.assertEqual(len(dataset), 10)
        self.assertEqual(len(dataset.load_data_list()), 10)
        self.assertEqual(dataset[0]['img'].shape, (26, 67, 3))
        self.assertEqual(dataset[0]['instances'][0]['text'], 'GRAND')
        self.assertEqual(dataset[1]['img'].shape, (17, 37, 3))
        self.assertEqual(dataset[1]['instances'][0]['text'], 'HOTEL')

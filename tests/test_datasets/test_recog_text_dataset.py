# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from mmocr.datasets import RecogTextDataset


class TestRecogTextDataset(TestCase):

    def test_txt_dataset(self):

        # test initialization
        dataset = RecogTextDataset(
            ann_file='tests/data/rec_toy_dataset/old_label.txt',
            data_prefix=dict(img_path='imgs'),
            parser_cfg=dict(
                type='LineStrParser',
                keys=['filename', 'text'],
                keys_idx=[0, 1]),
            pipeline=[])
        dataset.full_init()
        self.assertEqual(len(dataset), 10)
        self.assertEqual(len(dataset.load_data_list()), 10)

        # test load_data_list
        anno = dataset.load_data_list()
        self.assertIn(anno[0]['img_path'],
                      ['imgs/1223731.jpg', 'imgs\\1223731.jpg'])
        self.assertEqual(anno[0]['instances'][0]['text'], 'GRAND')
        self.assertIn(anno[1]['img_path'],
                      ['imgs/1223733.jpg', 'imgs\\1223733.jpg'])

        self.assertEqual(anno[1]['instances'][0]['text'], 'HOTEL')

    def test_jsonl_dataset(self):
        dataset = RecogTextDataset(
            ann_file='tests/data/rec_toy_dataset/old_label.jsonl',
            data_prefix=dict(img_path='imgs'),
            parser_cfg=dict(type='LineJsonParser', keys=['filename', 'text']),
            pipeline=[])
        dataset.full_init()
        self.assertEqual(len(dataset), 10)
        self.assertEqual(len(dataset.load_data_list()), 10)

        # test load_data_list
        anno = dataset.load_data_list()
        self.assertIn(anno[0]['img_path'],
                      ['imgs/1223731.jpg', 'imgs\\1223731.jpg'])
        self.assertEqual(anno[0]['instances'][0]['text'], 'GRAND')
        self.assertIn(anno[1]['img_path'],
                      ['imgs/1223733.jpg', 'imgs\\1223733.jpg'])
        self.assertEqual(anno[1]['instances'][0]['text'], 'HOTEL')

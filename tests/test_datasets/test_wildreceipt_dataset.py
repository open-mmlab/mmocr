# Copyright (c) OpenMMLab. All rights reserved.
import unittest

from mmocr.datasets import WildReceiptDataset


class TestWildReceiptDataset(unittest.TestCase):

    def setUp(self):
        metainfo = 'tests/data/kie_toy_dataset/wildreceipt/class_list.txt'
        self.dataset = WildReceiptDataset(
            data_prefix=dict(img_path='data/'),
            ann_file='tests/data/kie_toy_dataset/wildreceipt/data.txt',
            metainfo=metainfo,
            pipeline=[],
            serialize_data=False,
            lazy_init=False)

    def test_init(self):
        self.assertEqual(self.dataset.metainfo['category'][0], {
            'id': '0',
            'name': 'Ignore'
        })
        self.assertEqual(self.dataset.metainfo['task_name'], 'KIE')
        self.assertEqual(self.dataset.metainfo['dataset_type'],
                         'WildReceiptDataset')

    def test_getitem(self):
        data = self.dataset.__getitem__(0)

        instance = data['instances'][0]
        self.assertIsInstance(instance['bbox_label'], int)
        self.assertIsInstance(instance['edge_label'], int)
        self.assertIsInstance(instance['text'], str)
        self.assertEqual(instance['bbox'].shape, (4, ))
        self.assertEqual(data['img_shape'], (1200, 1600))
        self.assertEqual(
            data['img_path'],
            'data/image_files/Image_16/11/d5de7f2a20751e50b84c747c17a24cd98bed3554.jpeg'  # noqa
        )

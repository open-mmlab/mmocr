# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from copy import deepcopy
from unittest import TestCase
from unittest.mock import MagicMock

from mmengine.registry import init_default_scope

from mmocr.datasets import ConcatDataset, OCRDataset
from mmocr.registry import TRANSFORMS


class TestConcatDataset(TestCase):

    @TRANSFORMS.register_module()
    class MockTransform:

        def __init__(self, return_value):
            self.return_value = return_value

        def __call__(self, *args, **kwargs):
            return self.return_value

    def setUp(self):

        init_default_scope('mmocr')
        dataset = OCRDataset

        # create dataset_a
        data_info = dict(filename='img_1.jpg', height=720, width=1280)
        dataset.parse_data_info = MagicMock(return_value=data_info)

        self.dataset_a = dataset(
            data_root=osp.join(
                osp.dirname(__file__), '../data/det_toy_dataset'),
            data_prefix=dict(img_path='imgs'),
            ann_file='instances_test.json')

        self.dataset_a_with_pipeline = dataset(
            data_root=osp.join(
                osp.dirname(__file__), '../data/det_toy_dataset'),
            data_prefix=dict(img_path='imgs'),
            ann_file='instances_test.json',
            pipeline=[dict(type='MockTransform', return_value=1)])

        # create dataset_b
        data_info = dict(filename='img_2.jpg', height=720, width=1280)
        dataset.parse_data_info = MagicMock(return_value=data_info)
        self.dataset_b = dataset(
            data_root=osp.join(
                osp.dirname(__file__), '../data/det_toy_dataset'),
            data_prefix=dict(img_path='imgs'),
            ann_file='instances_test.json')
        self.dataset_b_with_pipeline = dataset(
            data_root=osp.join(
                osp.dirname(__file__), '../data/det_toy_dataset'),
            data_prefix=dict(img_path='imgs'),
            ann_file='instances_test.json',
            pipeline=[dict(type='MockTransform', return_value=2)])

    def test_init(self):
        with self.assertRaises(TypeError):
            ConcatDataset(datasets=[0])
        with self.assertRaises(ValueError):
            ConcatDataset(
                datasets=[
                    deepcopy(self.dataset_a_with_pipeline),
                    deepcopy(self.dataset_b)
                ],
                pipeline=[dict(type='MockTransform', return_value=3)])

        with self.assertRaises(ValueError):
            ConcatDataset(
                datasets=[
                    deepcopy(self.dataset_a),
                    deepcopy(self.dataset_b_with_pipeline)
                ],
                pipeline=[dict(type='MockTransform', return_value=3)])
        with self.assertRaises(ValueError):
            dataset_a = deepcopy(self.dataset_a)
            dataset_b = OCRDataset(
                metainfo=dict(dummy='dummy'),
                data_root=osp.join(
                    osp.dirname(__file__), '../data/det_toy_dataset'),
                data_prefix=dict(img_path='imgs'),
                ann_file='instances_test.json')
            ConcatDataset(datasets=[dataset_a, dataset_b])
        # test lazy init
        ConcatDataset(
            datasets=[deepcopy(self.dataset_a),
                      deepcopy(self.dataset_b)],
            pipeline=[dict(type='MockTransform', return_value=3)],
            lazy_init=True)

    def test_getitem(self):
        cat_datasets = ConcatDataset(
            datasets=[deepcopy(self.dataset_a),
                      deepcopy(self.dataset_b)],
            pipeline=[dict(type='MockTransform', return_value=3)])
        for datum in cat_datasets:
            self.assertEqual(datum, 3)

        cat_datasets = ConcatDataset(
            datasets=[
                deepcopy(self.dataset_a_with_pipeline),
                deepcopy(self.dataset_b)
            ],
            pipeline=[dict(type='MockTransform', return_value=3)],
            force_apply=True)
        for datum in cat_datasets:
            self.assertEqual(datum, 3)

        cat_datasets = ConcatDataset(datasets=[
            deepcopy(self.dataset_a_with_pipeline),
            deepcopy(self.dataset_b_with_pipeline)
        ])
        self.assertEqual(cat_datasets[0], 1)
        self.assertEqual(cat_datasets[-1], 2)

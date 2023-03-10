# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import unittest

import cv2
import numpy as np

from mmocr.datasets.preparers import TextRecogCropPacker, TextRecogPacker


class TestTextRecogPacker(unittest.TestCase):

    def test_pack_instance(self):

        packer = TextRecogPacker(data_root='', split='test')
        sample = ('test.jpg', 'text')
        results = packer.pack_instance(sample)
        self.assertDictEqual(
            results,
            dict(
                img_path=osp.join('textrecog_imgs', 'test', 'test.jpg'),
                instances=[dict(text='text')]))

    def test_add_meta(self):
        packer = TextRecogPacker(data_root='', split='test')
        sample = [dict(img_path='test.jpg', instances=[dict(text='text')])]
        results = packer.add_meta(sample)
        self.assertDictEqual(
            results,
            dict(
                metainfo=dict(
                    dataset_type='TextRecogDataset', task_name='textrecog'),
                data_list=sample))


class TestTextRecogCropPacker(unittest.TestCase):

    def setUp(self) -> None:
        self.root = tempfile.TemporaryDirectory()
        img = np.random.randint(0, 255, (30, 40, 3), dtype=np.uint8)
        cv2.imwrite(osp.join(self.root.name, 'test_img.jpg'), img)
        self.instance = [{
            'poly': [0, 0, 0, 10, 10, 20, 20, 0],
            'ignore': False,
            'text': 'text1'
        }, {
            'box': [0, 0, 10, 20],
            'ignore': False,
            'text': 'text2'
        }]
        self.img_path = osp.join(self.root.name, 'test_img.jpg')
        self.sample = (self.img_path, self.instance)

    def test_pack_instance(self):
        packer = TextRecogCropPacker(data_root=self.root.name, split='test')
        instance = packer.pack_instance(self.sample)
        self.assertListEqual(instance, [
            dict(
                img_path=osp.join('textrecog_imgs', 'test', 'test_img_0.jpg'),
                instances=[dict(text='text1')]),
            dict(
                img_path=osp.join('textrecog_imgs', 'test', 'test_img_1.jpg'),
                instances=[dict(text='text2')])
        ])

    def test_add_meta(self):
        packer = TextRecogCropPacker(data_root=self.root.name, split='test')
        instance = packer.pack_instance(self.sample)
        results = packer.add_meta([instance])
        self.assertDictEqual(
            results,
            dict(
                metainfo=dict(
                    dataset_type='TextRecogDataset', task_name='textrecog'),
                data_list=instance))

    def tearDown(self) -> None:
        self.root.cleanup()

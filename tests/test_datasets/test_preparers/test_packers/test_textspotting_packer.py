# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import unittest

import cv2
import numpy as np

from mmocr.datasets.preparers import TextSpottingPacker


class TestTextSpottingPacker(unittest.TestCase):

    def setUp(self) -> None:
        self.root = tempfile.TemporaryDirectory()
        img = np.random.randint(0, 255, (30, 20, 3), dtype=np.uint8)
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
        packer = TextSpottingPacker(data_root=self.root.name, split='test')
        instance = packer.pack_instance(self.sample)
        self.assertEquals(instance['img_path'], 'test_img.jpg')
        self.assertEquals(instance['height'], 30)
        self.assertEquals(instance['width'], 20)
        self.assertEquals(instance['instances'][0]['polygon'],
                          [0, 0, 0, 10, 10, 20, 20, 0])
        self.assertEquals(instance['instances'][0]['bbox'],
                          [float(x) for x in [0, 0, 20, 20]])
        self.assertEquals(instance['instances'][0]['bbox_label'], 0)
        self.assertEquals(instance['instances'][0]['ignore'], False)
        self.assertEquals(instance['instances'][0]['text'], 'text1')
        self.assertEquals(instance['instances'][1]['polygon'],
                          [0.0, 0.0, 10.0, 0.0, 10.0, 20.0, 0.0, 20.0])
        self.assertEquals(instance['instances'][1]['bbox'],
                          [float(x) for x in [0, 0, 10, 20]])
        self.assertEquals(instance['instances'][1]['bbox_label'], 0)
        self.assertEquals(instance['instances'][1]['ignore'], False)
        self.assertEquals(instance['instances'][1]['text'], 'text2')

    def test_add_meta(self):
        packer = TextSpottingPacker(data_root=self.root.name, split='test')
        instance = packer.pack_instance(self.sample)
        meta = packer.add_meta(instance)
        self.assertDictEqual(
            meta, {
                'metainfo': {
                    'dataset_type': 'TextSpottingDataset',
                    'task_name': 'textspotting',
                    'category': [{
                        'id': 0,
                        'name': 'text'
                    }]
                },
                'data_list': instance
            })

    def tearDown(self) -> None:
        self.root.cleanup()

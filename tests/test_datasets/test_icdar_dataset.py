# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from unittest import TestCase

import mmengine

from mmocr.datasets.icdar_dataset import IcdarDataset


class TestIcdarDataset(TestCase):

    def _create_dummy_icdar_json(self, json_name):
        image_1 = {
            'id': 0,
            'width': 640,
            'height': 640,
            'file_name': 'fake_name.jpg',
        }
        image_2 = {
            'id': 1,
            'width': 640,
            'height': 640,
            'file_name': 'fake_name1.jpg',
        }

        annotation_1 = {
            'id': 1,
            'image_id': 0,
            'category_id': 0,
            'area': 400,
            'bbox': [50, 60, 20, 20],
            'iscrowd': 0,
            'segmentation': [[50, 60, 70, 60, 70, 80, 50, 80]]
        }

        annotation_2 = {
            'id': 2,
            'image_id': 0,
            'category_id': 0,
            'area': 900,
            'bbox': [100, 120, 30, 30],
            'iscrowd': 0,
            'segmentation': [[100, 120, 130, 120, 120, 150, 100, 150]]
        }

        annotation_3 = {
            'id': 3,
            'image_id': 0,
            'category_id': 0,
            'area': 1600,
            'bbox': [150, 160, 40, 40],
            'iscrowd': 1,
            'segmentation': [[150, 160, 190, 160, 190, 200, 150, 200]]
        }

        annotation_4 = {
            'id': 4,
            'image_id': 0,
            'category_id': 0,
            'area': 10000,
            'bbox': [250, 260, 100, 100],
            'iscrowd': 1,
            'segmentation': [[250, 260, 350, 260, 350, 360, 250, 360]]
        }
        annotation_5 = {
            'id': 5,
            'image_id': 1,
            'category_id': 0,
            'area': 10000,
            'bbox': [250, 260, 100, 100],
            'iscrowd': 1,
            'segmentation': [[250, 260, 350, 260, 350, 360, 250, 360]]
        }
        annotation_6 = {
            'id': 6,
            'image_id': 1,
            'category_id': 0,
            'area': 0,
            'bbox': [0, 0, 0, 0],
            'iscrowd': 1,
            'segmentation': [[250, 260, 350, 260, 350, 360, 250, 360]]
        }
        annotation_7 = {
            'id': 7,
            'image_id': 1,
            'category_id': 2,
            'area': 10000,
            'bbox': [250, 260, 100, 100],
            'iscrowd': 1,
            'segmentation': [[250, 260, 350, 260, 350, 360, 250, 360]]
        }
        annotation_8 = {
            'id': 8,
            'image_id': 1,
            'category_id': 0,
            'area': 10000,
            'bbox': [250, 260, 100, 100],
            'iscrowd': 1,
            'segmentation': [[250, 260, 350, 260, 350, 360, 250, 360]]
        }

        categories = [{
            'id': 0,
            'name': 'text',
            'supercategory': 'text',
        }]

        fake_json = {
            'images': [image_1, image_2],
            'annotations': [
                annotation_1, annotation_2, annotation_3, annotation_4,
                annotation_5, annotation_6, annotation_7, annotation_8
            ],
            'categories':
            categories
        }
        self.metainfo = dict(CLASSES=('text'))
        mmengine.dump(fake_json, json_name)

    def test_icdar_dataset(self):
        tmp_dir = tempfile.TemporaryDirectory()
        # create dummy data
        fake_json_file = osp.join(tmp_dir.name, 'fake_data.json')
        self._create_dummy_icdar_json(fake_json_file)

        # test initialization
        dataset = IcdarDataset(
            ann_file=fake_json_file,
            data_prefix=dict(img_path='imgs'),
            metainfo=self.metainfo,
            pipeline=[])
        self.assertEqual(dataset.metainfo['CLASSES'], self.metainfo['CLASSES'])
        dataset.full_init()
        self.assertEqual(len(dataset), 2)
        self.assertEqual(len(dataset.load_data_list()), 2)

        # test load_data_list
        anno = dataset.load_data_list()[0]
        self.assertEqual(len(anno['instances']), 4)
        self.assertTrue('ignore' in anno['instances'][0])
        self.assertTrue('bbox' in anno['instances'][0])
        self.assertEqual(anno['instances'][0]['bbox_label'], 0)
        self.assertTrue('polygon' in anno['instances'][0])
        tmp_dir.cleanup()

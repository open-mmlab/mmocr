# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from unittest import TestCase

import mmengine
from mmocr.utils.data_converter_utils import (dump_ocr_data,
                                              recog_anno_to_imginfo)


class TestDataConverterUtils(TestCase):

    def _create_dummy_data(self):
        img_info = dict(
            file_name='test.jpg', height=100, width=200, segm_file='seg.txt')
        anno_info = [
            dict(
                iscrowd=0,
                category_id=0,
                bbox=[0, 0, 10, 20],  # x, y, w, h
                text='t1',
                segmentation=[0, 0, 0, 10, 10, 20, 20, 0]),
            dict(
                iscrowd=1,
                category_id=0,
                bbox=[10, 10, 20, 20],  # x, y, w, h
                text='t2',
                segmentation=[10, 10, 10, 30, 30, 30, 30, 10]),
        ]
        img_info['anno_info'] = anno_info
        img_infos = [img_info]

        det_target = {
            'metainfo': {
                'dataset_type': 'TextDetDataset',
                'task_name': 'textdet',
                'category': [{
                    'id': 0,
                    'name': 'text'
                }],
            },
            'data_list': [{
                'img_path':
                'test.jpg',
                'height':
                100,
                'width':
                200,
                'seg_map':
                'seg.txt',
                'instances': [
                    {
                        'bbox': [0, 0, 10, 20],
                        'bbox_label': 0,
                        'polygon': [0, 0, 0, 10, 10, 20, 20, 0],
                        'ignore': False
                    },
                    {
                        'bbox': [10, 10, 30, 30],  # x1, y1, x2, y2
                        'bbox_label': 0,
                        'polygon': [10, 10, 10, 30, 30, 30, 30, 10],
                        'ignore': True
                    }
                ]
            }]
        }

        spotter_target = {
            'metainfo': {
                'dataset_type': 'TextSpotterDataset',
                'task_name': 'textspotter',
                'category': [{
                    'id': 0,
                    'name': 'text'
                }],
            },
            'data_list': [{
                'img_path':
                'test.jpg',
                'height':
                100,
                'width':
                200,
                'seg_map':
                'seg.txt',
                'instances': [
                    {
                        'bbox': [0, 0, 10, 20],
                        'bbox_label': 0,
                        'polygon': [0, 0, 0, 10, 10, 20, 20, 0],
                        'text': 't1',
                        'ignore': False
                    },
                    {
                        'bbox': [10, 10, 30, 30],  # x1, y1, x2, y2
                        'bbox_label': 0,
                        'polygon': [10, 10, 10, 30, 30, 30, 30, 10],
                        'text': 't2',
                        'ignore': True
                    }
                ]
            }]
        }

        recog_target = {
            'metainfo': {
                'dataset_type': 'TextRecogDataset',
                'task_name': 'textrecog',
            },
            'data_list': [{
                'img_path': 'test.jpg',
                'instances': [{
                    'text': 't1',
                }, {
                    'text': 't2',
                }]
            }]
        }

        return img_infos, det_target, spotter_target, recog_target

    def test_dump_ocr_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = osp.join(tmpdir, 'ocr.json')
            input_data, det_target, spotter_target, recog_target = \
                self._create_dummy_data()

            dump_ocr_data(input_data, output_path, 'textdet')
            result = mmengine.load(output_path)
            self.assertDictEqual(result, det_target)

            dump_ocr_data(input_data, output_path, 'textspotter')
            result = mmengine.load(output_path)
            self.assertDictEqual(result, spotter_target)

            dump_ocr_data(input_data, output_path, 'textrecog')
            result = mmengine.load(output_path)
            self.assertDictEqual(result, recog_target)

    def test_recog_anno_to_imginfo(self):
        file_paths = ['a.jpg', 'b.jpg']
        labels = ['aaa']
        with self.assertRaises(AssertionError):
            recog_anno_to_imginfo(file_paths, labels)

        file_paths = ['a.jpg', 'b.jpg']
        labels = ['aaa', 'bbb']
        target = [
            {
                'file_name': 'a.jpg',
                'anno_info': [{
                    'text': 'aaa'
                }]
            },
            {
                'file_name': 'b.jpg',
                'anno_info': [{
                    'text': 'bbb'
                }]
            },
        ]
        self.assertListEqual(target, recog_anno_to_imginfo(file_paths, labels))

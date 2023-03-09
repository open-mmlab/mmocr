# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
import tempfile
import unittest

from mmocr.datasets.preparers import NAFAnnParser


class TestNAFAnnParser(unittest.TestCase):

    def setUp(self) -> None:
        self.root = tempfile.TemporaryDirectory()

    def _create_fake_sample(self):
        fake_sample = {
            'fieldBBs': [{
                'poly_points': [[1357, 322], [1636, 324], [1636, 402],
                                [1357, 400]],
                'type':
                'field',
                'id':
                'f0',
                'isBlank':
                1
            }, {
                'poly_points': [[1831, 352], [1908, 353], [1908, 427],
                                [1830, 427]],
                'type':
                'blank',
                'id':
                'f1',
                'isBlank':
                1
            }],
            'textBBs': [{
                'poly_points': [[1388, 80], [2003, 82], [2003, 133],
                                [1388, 132]],
                'type':
                'text',
                'id':
                't0'
            }, {
                'poly_points': [[1065, 366], [1320, 366], [1320, 413],
                                [1065, 412]],
                'type':
                'text',
                'id':
                't1'
            }],
            'imageFilename':
            '004173988_00005.jpg',
            'transcriptions': {
                'f0': '7/24',
                'f1': '9',
                't0': 'REGISTRY RETURN RECEIPT.',
                't1': 'Date of delivery',
            }
        }
        ann_path = osp.join(self.root.name, 'naf.json')
        with open(ann_path, 'w') as f:
            json.dump(fake_sample, f)
        return ann_path

    def test_parsers(self):
        ann_path = self._create_fake_sample()
        parser = NAFAnnParser(split='train')
        _, instances = parser.parse_file('fake.jpg', ann_path)
        self.assertEqual(len(instances), 3)
        self.assertEqual(instances[0]['ignore'], False)
        self.assertEqual(instances[1]['ignore'], False)
        self.assertListEqual(instances[2]['poly'],
                             [1357, 322, 1636, 324, 1636, 402, 1357, 400])

        parser = NAFAnnParser(split='train', det=False)
        _, instances = parser.parse_file('fake.jpg', ann_path)
        self.assertEqual(len(instances), 2)
        self.assertEqual(instances[0]['text'], '7/24')

    def tearDown(self) -> None:
        self.root.cleanup()

# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
import tempfile
import unittest

from mmocr.datasets.preparers import FUNSDTextDetAnnParser


class TestFUNSDTextDetAnnParser(unittest.TestCase):

    def setUp(self) -> None:
        self.root = tempfile.TemporaryDirectory()

    def _create_fake_sample(self):
        fake_sample = {
            'form': [{
                'box': [91, 279, 123, 294],
                'text': 'Date:',
                'label': 'question',
                'words': [{
                    'box': [91, 279, 123, 294],
                    'text': 'Date:'
                }],
                'linking': [[0, 16]],
                'id': 0
            }, {
                'box': [92, 310, 130, 324],
                'text': 'From:',
                'label': 'question',
                'words': [{
                    'box': [92, 310, 130, 324],
                    'text': ''
                }],
                'linking': [[1, 22]],
                'id': 1
            }]
        }
        ann_path = osp.join(self.root.name, 'funsd.json')
        with open(ann_path, 'w') as f:
            json.dump(fake_sample, f)
        return ann_path

    def test_textdet_parsers(self):
        ann_path = self._create_fake_sample()
        parser = FUNSDTextDetAnnParser(split='train')
        _, instances = parser.parse_file('fake.jpg', ann_path)
        self.assertEqual(len(instances), 2)
        self.assertEqual(instances[0]['text'], 'Date:')
        self.assertEqual(instances[0]['ignore'], False)
        self.assertEqual(instances[1]['ignore'], True)
        self.assertListEqual(instances[0]['poly'],
                             [91, 279, 123, 279, 123, 294, 91, 294])

    def tearDown(self) -> None:
        self.root.cleanup()

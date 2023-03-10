# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
import tempfile
import unittest

from mmocr.datasets.preparers.parsers.wildreceipt_parser import (
    WildreceiptKIEAnnParser, WildreceiptTextDetAnnParser)
from mmocr.utils import list_to_file


class TestWildReceiptParsers(unittest.TestCase):

    def setUp(self) -> None:
        self.root = tempfile.TemporaryDirectory()
        fake_sample = dict(
            file_name='test.jpg',
            height=100,
            width=100,
            annotations=[
                dict(
                    box=[
                        550.0, 190.0, 937.0, 190.0, 937.0, 104.0, 550.0, 104.0
                    ],
                    text='test',
                    label=1,
                ),
                dict(
                    box=[
                        1048.0, 211.0, 1074.0, 211.0, 1074.0, 196.0, 1048.0,
                        196.0
                    ],
                    text='ATOREMGRTOMMILAZZO',
                    label=0,
                )
            ])
        fake_sample = [json.dumps(fake_sample)]
        self.anno = osp.join(self.root.name, 'wildreceipt.txt')
        list_to_file(self.anno, fake_sample)

    def test_textdet_parsers(self):
        parser = WildreceiptTextDetAnnParser(split='train')
        samples = parser.parse_files(self.root.name, self.anno)
        self.assertEqual(len(samples), 1)
        self.assertEqual(osp.basename(samples[0][0]), 'test.jpg')
        instances = samples[0][1]
        self.assertEqual(len(instances), 2)
        self.assertIn('poly', instances[0])
        self.assertIn('text', instances[0])
        self.assertIn('ignore', instances[0])
        self.assertEqual(instances[0]['text'], 'test')
        self.assertEqual(instances[1]['ignore'], True)

    def test_kie_parsers(self):
        parser = WildreceiptKIEAnnParser(split='train')
        samples = parser.parse_files(self.root.name, self.anno)
        self.assertEqual(len(samples), 1)

    def tearDown(self) -> None:
        self.root.cleanup()

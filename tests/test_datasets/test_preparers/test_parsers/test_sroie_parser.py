# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import unittest

from mmocr.datasets.preparers import SROIETextDetAnnParser
from mmocr.utils import list_to_file


class TestSROIETextDetAnnParser(unittest.TestCase):

    def setUp(self) -> None:
        self.root = tempfile.TemporaryDirectory()

    def _create_dummy_sroie_det(self):
        fake_anno = [
            '114,54,326,54,326,92,114,92,TAN CHAY YEE',
            '60,119,300,119,300,136,60,136,###',
            '100,139,267,139,267,162,100,162,ROC NO: 538358-H',
            '83,163,277,163,277,183,83,183,NO 2 & 4, JALAN BAYU 4,',
        ]
        ann_file = osp.join(self.root.name, 'sroie_det.txt')
        list_to_file(ann_file, fake_anno)
        return (osp.join(self.root.name, 'sroie_det.jpg'), ann_file)

    def test_textdet_parsers(self):
        file = self._create_dummy_sroie_det()
        parser = SROIETextDetAnnParser(split='train')

        img, instances = parser.parse_file(*file)
        self.assertEqual(img, file[0])
        self.assertEqual(len(instances), 4)
        self.assertIn('poly', instances[0])
        self.assertIn('text', instances[0])
        self.assertIn('ignore', instances[0])
        self.assertEqual(instances[0]['text'], 'TAN CHAY YEE')
        self.assertEqual(instances[1]['ignore'], True)
        self.assertEqual(instances[3]['text'], 'NO 2 & 4, JALAN BAYU 4,')
        self.assertListEqual(instances[2]['poly'],
                             [100, 139, 267, 139, 267, 162, 100, 162])

    def tearDown(self) -> None:
        self.root.cleanup()

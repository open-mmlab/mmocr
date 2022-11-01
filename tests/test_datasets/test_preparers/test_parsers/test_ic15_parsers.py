# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import unittest

from mmocr.datasets.preparers.parsers.ic15_parser import (
    ICDARTxtTextDetAnnParser, ICDARTxtTextRecogAnnParser)
from mmocr.utils import list_to_file


class TestIC15Parsers(unittest.TestCase):

    def setUp(self) -> None:
        self.root = tempfile.TemporaryDirectory()

    def _create_dummy_ic15_det(self):
        fake_anno = [
            '377,117,463,117,465,130,378,130,Genaxis Theatre',
            '493,115,519,115,519,131,493,131,[06]',
            '374,155,409,155,409,170,374,170,###',
        ]
        ann_file = osp.join(self.root.name, 'ic15_det.txt')
        list_to_file(ann_file, fake_anno)
        return (osp.join(self.root.name, 'ic15_det.jpg'), ann_file)

    def _create_dummy_ic15_recog(self):
        fake_anno = [
            'word_1.png, "Genaxis Theatre"',
            'word_2.png, "[06]"',
            'word_3.png, "62-03"',
        ]
        ann_file = osp.join(self.root.name, 'ic15_recog.txt')
        list_to_file(ann_file, fake_anno)
        return ann_file

    def test_textdet_parsers(self):
        parser = ICDARTxtTextDetAnnParser()
        file = self._create_dummy_ic15_det()
        img, instances = parser.parse_file(file, 'train')
        self.assertEqual(img, file[0])
        self.assertEqual(len(instances), 3)
        self.assertIn('poly', instances[0])
        self.assertIn('text', instances[0])
        self.assertIn('ignore', instances[0])
        self.assertEqual(instances[0]['text'], 'Genaxis Theatre')
        self.assertEqual(instances[2]['ignore'], True)

    def test_textrecog_parsers(self):
        parser = ICDARTxtTextRecogAnnParser()
        file = self._create_dummy_ic15_recog()
        samples = parser.parse_files(file, 'train')
        self.assertEqual(len(samples), 3)
        img, text = samples[0]
        self.assertEqual(img, 'word_1.png')
        self.assertEqual(text, 'Genaxis Theatre')

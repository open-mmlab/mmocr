# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import unittest

from mmocr.datasets.preparers.parsers.icdar_txt_parser import (
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
            '374,155,409,155,409,170,374,170,100,000', ' '
        ]
        ann_file = osp.join(self.root.name, 'ic15_det.txt')
        list_to_file(ann_file, fake_anno)
        return (osp.join(self.root.name, 'ic15_det.jpg'), ann_file)

    def _create_dummy_ic15_recog(self):
        fake_anno = [
            'word_1.png, "Genaxis Theatre"', 'word_2.png, "[06]"',
            'word_3.png, "62-03"', 'word_4.png, "62-,03"', ''
        ]
        ann_file = osp.join(self.root.name, 'ic15_recog.txt')
        list_to_file(ann_file, fake_anno)
        return ann_file

    def test_textdet_parsers(self):
        file = self._create_dummy_ic15_det()
        parser = ICDARTxtTextDetAnnParser(split='train')

        img, instances = parser.parse_file(*file)
        self.assertEqual(img, file[0])
        self.assertEqual(len(instances), 4)
        self.assertIn('poly', instances[0])
        self.assertIn('text', instances[0])
        self.assertIn('ignore', instances[0])
        self.assertEqual(instances[0]['text'], 'Genaxis Theatre')
        self.assertEqual(instances[2]['ignore'], True)
        self.assertEqual(instances[3]['text'], '100,000')

    def test_textrecog_parsers(self):
        parser = ICDARTxtTextRecogAnnParser(split='train')
        file = self._create_dummy_ic15_recog()
        samples = parser.parse_files(self.root.name, file)
        self.assertEqual(len(samples), 4)
        img, text = samples[0]
        self.assertEqual(img, osp.join(self.root.name, 'word_1.png'))
        self.assertEqual(text, 'Genaxis Theatre')
        img, text = samples[3]
        self.assertEqual(text, '62-,03')

    def tearDown(self) -> None:
        self.root.cleanup()

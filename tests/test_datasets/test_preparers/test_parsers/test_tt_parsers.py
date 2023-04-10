# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import unittest

from mmocr.datasets.preparers.parsers.totaltext_parser import \
    TotaltextTextDetAnnParser
from mmocr.utils import list_to_file


class TestTTParsers(unittest.TestCase):

    def setUp(self) -> None:
        self.root = tempfile.TemporaryDirectory()

    def _create_dummy_tt_det(self):
        fake_anno = [
            "x: [[ 53 120 121  56]], y: [[446 443 456 458]], ornt: [u'h'], transcriptions: [u'PERUNDING']",  # noqa: E501
            "x: [[123 165 166 125]], y: [[443 440 453 455]], ornt: [u'h'], transcriptions: [u'PENILAI']",  # noqa: E501
            "x: [[168 179 179 167]], y: [[439 439 452 453]], ornt: [u'#'], transcriptions: [u'#']",  # noqa: E501
        ]
        ann_file = osp.join(self.root.name, 'tt_det.txt')
        list_to_file(ann_file, fake_anno)
        return (osp.join(self.root.name, 'tt_det.jpg'), ann_file)

    def test_textdet_parsers(self):
        parser = TotaltextTextDetAnnParser(split='train')
        file = self._create_dummy_tt_det()
        img, instances = parser.parse_file(*file)
        self.assertEqual(img, file[0])
        self.assertEqual(len(instances), 3)
        self.assertIn('poly', instances[0])
        self.assertIn('text', instances[0])
        self.assertIn('ignore', instances[0])
        self.assertEqual(instances[0]['text'], 'PERUNDING')
        self.assertEqual(instances[2]['ignore'], True)

    def tearDown(self) -> None:
        self.root.cleanup()

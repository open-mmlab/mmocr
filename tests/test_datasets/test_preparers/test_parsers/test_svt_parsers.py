# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import unittest

from mmocr.datasets.preparers.parsers.svt_parser import SVTTextDetAnnParser
from mmocr.utils import list_to_file


class TestSVTParsers(unittest.TestCase):

    def setUp(self) -> None:
        self.root = tempfile.TemporaryDirectory()

    def _create_dummy_svt_det(self):
        fake_anno = [
            '<?xml version="1.0" encoding="utf-8"?>',
            '<tagset>',
            '   <image>',
            '      <imageName>img/test.jpg</imageName>',
            '      <Resolution x="1280" y="880"/>',
            '      <taggedRectangles>',
            '         <taggedRectangle height="75" width="236" x="375" y="253">',  # noqa: E501
            '            <tag>LIVING</tag>',
            '         </taggedRectangle>',
            '         <taggedRectangle height="76" width="175" x="639" y="272">',  # noqa: E501
            '            <tag>ROOM</tag>',
            '         </taggedRectangle>',
            '         <taggedRectangle height="87" width="281" x="839" y="283">',  # noqa: E501
            '            <tag>THEATERS</tag>',
            '         </taggedRectangle>',
            '      </taggedRectangles>',
            '   </image>',
            '</tagset>',
        ]
        ann_file = osp.join(self.root.name, 'svt_det.xml')
        list_to_file(ann_file, fake_anno)
        return ann_file

    def test_textdet_parsers(self):
        parser = SVTTextDetAnnParser(split='train')
        file = self._create_dummy_svt_det()
        samples = parser.parse_files(self.root.name, file)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0][0], osp.join(self.root.name, 'test.jpg'))
        self.assertEqual(len(samples[0][1]), 3)
        self.assertEqual(samples[0][1][0]['text'], 'living')
        self.assertEqual(samples[0][1][1]['text'], 'room')
        self.assertEqual(samples[0][1][2]['text'], 'theaters')
        self.assertEqual(samples[0][1][0]['poly'],
                         [375, 253, 611, 253, 611, 328, 375, 328])
        self.assertEqual(samples[0][1][0]['ignore'], False)

    def tearDown(self) -> None:
        self.root.cleanup()

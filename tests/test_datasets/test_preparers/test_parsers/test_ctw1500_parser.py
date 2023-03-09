# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import unittest

from mmocr.datasets.preparers.parsers import CTW1500AnnParser
from mmocr.utils import list_to_file


class TestCTW1500AnnParser(unittest.TestCase):

    def setUp(self) -> None:
        self.root = tempfile.TemporaryDirectory()

    def _create_dummy_ctw1500_det(self):
        fake_train_anno = [
            '<Annotations>',
            ' <image file="0200.jpg">',
            '   <box height="197" left="131" top="49" width="399">',
            '     <label>OLATHE</label>',
            '     <segs>131,58,208,49,279,56,346,76,412,101,473,141,530,192,510,246,458,210,405,175,350,151,291,137,228,133,165,134</segs>',  # noqa: E501
            '     <pts x="183" y="95" />',
            '     <pts x="251" y="89" />',
            '     <pts x="322" y="107" />',
            '     <pts x="383" y="124" />',
            '     <pts x="441" y="161" />',
            '     <pts x="493" y="201" />',
            '   </box>',
            ' </image>',
            '</Annotations>',
        ]
        train_ann_file = osp.join(self.root.name, 'ctw1500_train.xml')
        list_to_file(train_ann_file, fake_train_anno)

        fake_test_anno = [
            '48,84,61,79,75,73,88,68,102,74,116,79,130,84,135,73,119,67,104,60,89,56,74,61,59,67,45,73,#######',  # noqa: E501
            '51,137,58,137,66,137,74,137,82,137,90,137,98,137,98,119,90,119,82,119,74,119,66,119,58,119,50,119,####E-313',  # noqa: E501
            '41,155,49,155,57,155,65,155,73,155,81,155,89,155,87,136,79,136,71,136,64,136,56,136,48,136,41,137,#######',  # noqa: E501
            '41,193,57,193,74,194,90,194,107,195,123,195,140,196,146,168,128,167,110,167,92,167,74,166,56,166,39,166,####F.D.N.Y.',  # noqa: E501
        ]
        test_ann_file = osp.join(self.root.name, 'ctw1500_test.txt')
        list_to_file(test_ann_file, fake_test_anno)
        return (osp.join(self.root.name,
                         'ctw1500.jpg'), train_ann_file, test_ann_file)

    def test_textdet_parsers(self):
        parser = CTW1500AnnParser(split='train')
        img_path, train_file, test_file = self._create_dummy_ctw1500_det()
        img_path, instances = parser.parse_file(img_path, train_file)
        self.assertEqual(img_path, osp.join(self.root.name, 'ctw1500.jpg'))
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0]['text'], 'OLATHE')
        self.assertEqual(instances[0]['poly'], [
            131, 58, 208, 49, 279, 56, 346, 76, 412, 101, 473, 141, 530, 192,
            510, 246, 458, 210, 405, 175, 350, 151, 291, 137, 228, 133, 165,
            134
        ])
        self.assertEqual(instances[0]['ignore'], False)

        parser = CTW1500AnnParser(split='test')
        img_path, instances = parser.parse_file(img_path, test_file)
        self.assertEqual(img_path, osp.join(self.root.name, 'ctw1500.jpg'))
        self.assertEqual(len(instances), 4)
        self.assertEqual(instances[0]['ignore'], True)
        self.assertEqual(instances[1]['text'], 'E-313')
        self.assertEqual(instances[3]['poly'], [
            41, 193, 57, 193, 74, 194, 90, 194, 107, 195, 123, 195, 140, 196,
            146, 168, 128, 167, 110, 167, 92, 167, 74, 166, 56, 166, 39, 166
        ])

    def tearDown(self) -> None:
        self.root.cleanup()

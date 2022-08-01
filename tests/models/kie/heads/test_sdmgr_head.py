# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from unittest import TestCase

import torch
from mmengine import InstanceData

from mmocr.models.kie.heads import SDMGRHead
from mmocr.models.textrecog.dictionary import Dictionary
from mmocr.structures import KIEDataSample
from mmocr.testing import create_dummy_dict_file


class TestSDMGRHead(TestCase):

    def test_init(self):
        with self.assertRaises(AssertionError):
            SDMGRHead(dictionary='str')

    def test_forward(self):

        data_sample = KIEDataSample()
        data_sample.gt_instances = InstanceData(
            bboxes=torch.rand((2, 4)), texts=['t1', 't2'])
        with tempfile.TemporaryDirectory() as tmp_dir:
            dict_file = osp.join(tmp_dir, 'fake_chars.txt')
            create_dummy_dict_file(dict_file)
            dict_cfg = dict(
                type='Dictionary',
                dict_file=dict_file,
                with_unknown=True,
                with_padding=True,
                unknown_token=None)

            # Test img + dict_cfg
            head = SDMGRHead(dictionary=dict_cfg)
            node_cls, edge_cls = head(torch.rand((2, 64)), [data_sample])
            self.assertEqual(node_cls.shape, torch.Size([2, 26]))
            self.assertEqual(edge_cls.shape, torch.Size([4, 2]))

            # When input image is None
            head = SDMGRHead(dictionary=Dictionary(**dict_cfg))
            node_cls, edge_cls = head(None, [data_sample])
            self.assertEqual(node_cls.shape, torch.Size([2, 26]))
            self.assertEqual(edge_cls.shape, torch.Size([4, 2]))

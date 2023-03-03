# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import unittest

from mmocr.datasets.preparers.gatherers import MonoGatherer


class TestMonoGatherer(unittest.TestCase):

    def test_mono_text_gatherer(self):
        data_root = 'dummpy'
        img_dir = 'dummy_img'
        ann_dir = 'dummy_ann'
        ann_name = 'dummy_ann.json'
        split = 'train'
        gatherer = MonoGatherer(
            data_root=data_root,
            img_dir=img_dir,
            ann_dir=ann_dir,
            ann_name=ann_name,
            split=split)
        gather_img_dir, ann_path = gatherer()
        self.assertEqual(gather_img_dir, osp.join(data_root, img_dir))
        self.assertEqual(ann_path, osp.join(data_root, ann_dir, ann_name))

# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
import unittest

import cv2
import numpy as np

from mmocr.datasets.preparers.gatherers import PairGatherer


class TestPairGatherer(unittest.TestCase):

    def test_pair_text_gatherer(self):
        root = tempfile.TemporaryDirectory()
        data_root = root.name
        img_dir = 'dummy_img'
        ann_dir = 'dummy_ann'
        split = 'train'
        img = np.random.randint(0, 100, size=(100, 100, 3))
        os.makedirs(osp.join(data_root, img_dir))
        os.makedirs(osp.join(data_root, ann_dir))
        for i in range(10):
            cv2.imwrite(osp.join(data_root, img_dir, f'img_{i}.jpg'), img)
            f = open(osp.join(data_root, ann_dir, f'img_{i}.txt'), 'w')
            f.close()
        f = open(osp.join(data_root, ann_dir, 'img_10.mmocr'), 'w')
        f.close()
        gatherer = PairGatherer(
            data_root=data_root,
            img_dir=img_dir,
            ann_dir=ann_dir,
            split=split,
            img_suffixes=['.jpg'],
            rule=[r'img_(\d+)\.([jJ][pP][gG])', r'img_\1.txt'])
        img_list, ann_list = gatherer()
        self.assertEqual(len(img_list), 10)
        self.assertEqual(len(ann_list), 10)
        self.assertNotIn(
            osp.join(data_root, ann_dir, 'img_10.mmocr'), ann_list)
        root.cleanup()

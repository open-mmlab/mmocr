# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import shutil
import time
from unittest import TestCase
from unittest.mock import Mock

import torch

from mmengine.structures import InstanceData
from mmocr.engine.hooks import VisualizationHook
from mmocr.structures import TextDetDataSample
from mmocr.visualization import TextDetLocalVisualizer


def _rand_bboxes(num_boxes, h, w):
    cx, cy, bw, bh = torch.rand(num_boxes, 4).T

    tl_x = ((cx * w) - (w * bw / 2)).clamp(0, w).unsqueeze(0)
    tl_y = ((cy * h) - (h * bh / 2)).clamp(0, h).unsqueeze(0)
    br_x = ((cx * w) + (w * bw / 2)).clamp(0, w).unsqueeze(0)
    br_y = ((cy * h) + (h * bh / 2)).clamp(0, h).unsqueeze(0)

    bboxes = torch.cat([tl_x, tl_y, br_x, br_y], dim=0).T
    return bboxes


class TestVisualizationHook(TestCase):

    def setUp(self) -> None:

        data_sample = TextDetDataSample()
        data_sample.set_metainfo({
            'img_path':
            osp.join(
                osp.dirname(__file__),
                '../../data/det_toy_dataset/imgs/test/img_1.jpg')
        })

        pred_instances = InstanceData()
        pred_instances.bboxes = _rand_bboxes(5, 10, 12)
        pred_instances.labels = torch.randint(0, 2, (5, ))
        pred_instances.scores = torch.rand((5, ))

        data_sample.pred_instances = pred_instances
        self.outputs = [data_sample] * 2
        self.data_batch = None

    def test_after_val_iter(self):
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        TextDetLocalVisualizer.get_instance(
            'visualizer_val',
            vis_backends=[dict(type='LocalVisBackend', img_save_dir='')],
            save_dir=timestamp)
        runner = Mock()
        runner.iter = 1
        hook = VisualizationHook(enable=True, interval=1)
        self.assertFalse(osp.exists(timestamp))
        hook.after_val_iter(runner, 1, self.data_batch, self.outputs)
        self.assertTrue(osp.exists(timestamp))
        shutil.rmtree(timestamp)

    def test_after_test_iter(self):
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        TextDetLocalVisualizer.get_instance(
            'visualizer_test',
            vis_backends=[dict(type='LocalVisBackend', img_save_dir='')],
            save_dir=timestamp)
        runner = Mock()
        runner.iter = 1

        hook = VisualizationHook(enable=False)
        hook.after_test_iter(runner, 1, self.data_batch, self.outputs)
        self.assertFalse(osp.exists(timestamp))

        hook = VisualizationHook(enable=True)
        hook.after_test_iter(runner, 1, self.data_batch, self.outputs)
        self.assertTrue(osp.exists(timestamp))
        shutil.rmtree(timestamp)

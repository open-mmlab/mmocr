# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import shutil
import time
from unittest import TestCase
from unittest.mock import Mock

import torch
from mmengine.data import InstanceData

from mmocr.data import TextDetDataSample
from mmocr.engine.hooks import VisualizationHook
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
        TextDetLocalVisualizer.get_instance('visualizer')

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
        runner = Mock()
        runner.iter = 1
        hook = VisualizationHook(draw=True, interval=1)
        hook.after_val_iter(runner, 1, self.data_batch, self.outputs)

    def test_after_test_iter(self):
        runner = Mock()
        runner.iter = 1
        hook = VisualizationHook(draw=True)
        hook.after_test_iter(runner, 1, self.data_batch, self.outputs)

        # test
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        test_out_dir = timestamp + '1'
        runner.work_dir = timestamp
        runner.timestamp = '1'
        hook = VisualizationHook(draw=False, test_out_dir=test_out_dir)
        hook.after_test_iter(runner, 1, self.data_batch, self.outputs)
        self.assertTrue(not osp.exists(f'{timestamp}/1/{test_out_dir}'))

        hook = VisualizationHook(draw=True, test_out_dir=test_out_dir)
        hook.after_test_iter(runner, 1, self.data_batch, self.outputs)
        self.assertTrue(osp.exists(f'{timestamp}/1/{test_out_dir}'))
        shutil.rmtree(f'{timestamp}')

# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import unittest

import cv2
import numpy as np
import torch
from mmengine.data import InstanceData

from mmocr.core import TextDetDataSample
from mmocr.core.visualization import TextDetLocalVisualizer
from mmocr.utils import bbox2poly


class TestTextDetLocalVisualizer(unittest.TestCase):

    def setUp(self):
        h, w = 12, 10
        self.image = np.random.randint(0, 256, size=(h, w, 3)).astype('uint8')

        # gt_instances
        gt_det_data_sample = TextDetDataSample()
        gt_instances_data = dict(
            bboxes=self._rand_bboxes(5, h, w),
            polygons=self._rand_polys(5, h, w),
            labels=torch.zeros(5, ))
        gt_instances = InstanceData(**gt_instances_data)
        gt_det_data_sample.gt_instances = gt_instances
        self.gt_det_data_sample = gt_det_data_sample

        # pred_instances
        pred_det_data_sample = TextDetDataSample()
        pred_instances_data = dict(
            bboxes=self._rand_bboxes(5, h, w),
            polygons=self._rand_polys(5, h, w),
            labels=torch.zeros(5, ),
            scores=torch.rand((5, )))
        pred_instances = InstanceData(**pred_instances_data)
        pred_det_data_sample.pred_instances = pred_instances
        self.pred_det_data_sample = pred_det_data_sample

    def test_text_det_local_visualizer(self):
        for with_poly in [True, False]:
            for with_bbox in [True, False]:
                vis_cfg = dict(with_poly=with_poly, with_bbox=with_bbox)
                self._test_add_datasample(vis_cfg=vis_cfg)

    @staticmethod
    def _rand_bboxes(num_boxes, h, w):
        cx, cy, bw, bh = torch.rand(num_boxes, 4).T

        tl_x = ((cx * w) - (w * bw / 2)).clip(0, w)
        tl_y = ((cy * h) - (h * bh / 2)).clip(0, h)
        br_x = ((cx * w) + (w * bw / 2)).clip(0, w)
        br_y = ((cy * h) + (h * bh / 2)).clip(0, h)

        bboxes = torch.vstack([tl_x, tl_y, br_x, br_y]).T

        return bboxes

    def _rand_polys(self, num_bboxes, h, w):
        bboxes = self._rand_bboxes(num_bboxes, h, w)
        bboxes = bboxes.tolist()
        polys = [bbox2poly(bbox) for bbox in bboxes]
        return polys

    def _test_add_datasample(self, vis_cfg):
        image = self.image
        h, w, c = image.shape
        gt_det_data_sample = self.gt_det_data_sample
        pred_det_data_sample = self.pred_det_data_sample

        det_local_visualizer = TextDetLocalVisualizer(**vis_cfg)
        det_local_visualizer.add_datasample('image', image, gt_det_data_sample)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # test out
            out_file = osp.join(tmp_dir, 'out_file.jpg')
            det_local_visualizer.add_datasample(
                'image', image, gt_det_data_sample, out_file=out_file)
            self._assert_image_and_shape(out_file, (h, w, c))

            det_local_visualizer.add_datasample(
                'image',
                image,
                gt_det_data_sample,
                pred_det_data_sample,
                out_file=out_file)
            self._assert_image_and_shape(out_file, (h, w * 2, c))

            det_local_visualizer.add_datasample(
                'image',
                image,
                gt_det_data_sample,
                pred_det_data_sample,
                draw_gt=False,
                out_file=out_file)
            self._assert_image_and_shape(out_file, (h, w, c))

            det_local_visualizer.add_datasample(
                'image',
                image,
                gt_det_data_sample,
                pred_det_data_sample,
                draw_pred=False,
                out_file=out_file)
            self._assert_image_and_shape(out_file, (h, w, c))

    def _assert_image_and_shape(self, out_file, out_shape):
        self.assertTrue(osp.exists(out_file))
        drawn_img = cv2.imread(out_file)
        self.assertTrue(drawn_img.shape == out_shape)

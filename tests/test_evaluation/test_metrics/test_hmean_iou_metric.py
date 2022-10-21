# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np
import torch
from mmengine.structures import InstanceData

from mmocr.evaluation import HmeanIOUMetric
from mmocr.structures import TextDetDataSample


class TestHmeanIOU(unittest.TestCase):

    def setUp(self):
        """Create dummy test data.

        We denote the polygons as the following.
        gt_polys: gt_a, gt_b, gt_c, gt_d_ignored
        pred_polys: pred_a, pred_b, pred_c, pred_d

        There are two pairs of matches: (gt_a, pred_a) and (gt_b, pred_b),
        because the IoU > threshold.

        gt_c and pred_c do not match any of the polygons.

        pred_d is ignored in the recall computation since it overlaps
        gt_d_ignored and the precision > ignore_precision_thr.
        """
        data_sample = TextDetDataSample()
        gt_instances = InstanceData()
        gt_instances.polygons = [
            torch.FloatTensor([0, 0, 1, 0, 1, 1, 0, 1]),
            torch.FloatTensor([2, 0, 3, 0, 3, 1, 2, 1]),
            torch.FloatTensor([10, 0, 11, 0, 11, 1, 10, 1]),
            torch.FloatTensor([1, 0, 2, 0, 2, 1, 1, 1]),
        ]
        gt_instances.ignored = np.bool_([False, False, False, True])
        pred_instances = InstanceData()
        pred_instances.polygons = [
            torch.FloatTensor([0, 0, 1, 0, 1, 1, 0, 1]),
            torch.FloatTensor([2, 0.1, 3, 0.1, 3, 1.1, 2, 1.1]),
            torch.FloatTensor([1, 1, 2, 1, 2, 2, 1, 2]),
            torch.FloatTensor([1, -0.5, 2, -0.5, 2, 0.5, 1, 0.5]),
        ]
        pred_instances.scores = torch.FloatTensor([1, 1, 1, 0.001])
        data_sample.gt_instances = gt_instances
        data_sample.pred_instances = pred_instances
        predictions = [data_sample.to_dict()]

        data_sample = TextDetDataSample()
        gt_instances = InstanceData()
        gt_instances.polygons = [torch.FloatTensor([0, 0, 1, 0, 1, 1, 0, 1])]
        gt_instances.ignored = np.bool_([False])
        pred_instances = InstanceData()
        pred_instances.polygons = [
            torch.FloatTensor([0, 0, 1, 0, 1, 1, 0, 1]),
            torch.FloatTensor([0, 0, 1, 0, 1, 1, 0, 1])
        ]
        pred_instances.scores = torch.FloatTensor([1, 0.95])
        data_sample.gt_instances = gt_instances
        data_sample.pred_instances = pred_instances
        predictions.append(data_sample.to_dict())

        self.predictions = predictions

    def test_hmean_iou(self):

        metric = HmeanIOUMetric(prefix='mmocr')
        metric.process(None, self.predictions)
        eval_results = metric.evaluate(size=2)

        precision = 3 / 5
        recall = 3 / 4
        hmean = 2 * precision * recall / (precision + recall)
        target_result = {
            'precision': precision,
            'recall': recall,
            'hmean': hmean
        }
        self.assertDictEqual(target_result, eval_results)

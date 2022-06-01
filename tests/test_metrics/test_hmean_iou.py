# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np
import torch
from mmengine import InstanceData

from mmocr.core import TextDetDataSample
from mmocr.metrics import HmeanIOUMetric


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
        # prepare gt
        self.gt = [{
            'data_sample': {
                'instances': [{
                    'polygon': [0, 0, 1, 0, 1, 1, 0, 1],
                    'ignore': False
                }, {
                    'polygon': [2, 0, 3, 0, 3, 1, 2, 1],
                    'ignore': False
                }, {
                    'polygon': [10, 0, 11, 0, 11, 1, 10, 1],
                    'ignore': False
                }, {
                    'polygon': [1, 0, 2, 0, 2, 1, 1, 1],
                    'ignore': True
                }]
            }
        }, {
            'data_sample': {
                'instances': [{
                    'polygon': [0, 0, 1, 0, 1, 1, 0, 1],
                    'ignore': False
                }],
            }
        }]

        # prepare pred
        pred_data_sample = TextDetDataSample()
        pred_data_sample.pred_instances = InstanceData()
        pred_data_sample.pred_instances.polygons = [
            torch.FloatTensor([0, 0, 1, 0, 1, 1, 0, 1]),
            torch.FloatTensor([2, 0.1, 3, 0.1, 3, 1.1, 2, 1.1]),
            torch.FloatTensor([1, 1, 2, 1, 2, 2, 1, 2]),
            torch.FloatTensor([1, -0.5, 2, -0.5, 2, 0.5, 1, 0.5]),
        ]
        pred_data_sample.pred_instances.scores = torch.FloatTensor(
            [1, 1, 1, 0.001])
        predictions = [pred_data_sample.to_dict()]

        pred_data_sample = TextDetDataSample()
        pred_data_sample.pred_instances = InstanceData()
        pred_data_sample.pred_instances.polygons = [
            torch.FloatTensor([0, 0, 1, 0, 1, 1, 0, 1]),
            torch.FloatTensor([0, 0, 1, 0, 1, 1, 0, 1])
        ]
        pred_data_sample.pred_instances.scores = torch.FloatTensor([1, 0.95])
        predictions.append(pred_data_sample.to_dict())

        self.predictions = predictions

    def test_hmean_iou(self):

        metric = HmeanIOUMetric(prefix='mmocr')
        metric.process(self.gt, self.predictions)
        eval_results = metric.evaluate(size=2)

        precision = 3 / 5
        recall = 3 / 4
        hmean = 2 * precision * recall / (precision + recall)
        target_result = {
            'mmocr/precision': precision,
            'mmocr/recall': recall,
            'mmocr/hmean': hmean
        }
        self.assertDictEqual(target_result, eval_results)

    def test_compute_metrics(self):
        # Test different strategies
        fake_results = [
            dict(
                iou_metric=np.array([[1, 1], [1, 0]]),
                pred_scores=np.array([1., 1.]))
        ]

        # Vanilla
        metric = HmeanIOUMetric(strategy='vanilla')
        eval_results = metric.compute_metrics(fake_results)
        target_result = {'precision': 0.5, 'recall': 0.5, 'hmean': 0.5}
        self.assertDictEqual(target_result, eval_results)

        # Max matching
        metric = HmeanIOUMetric(strategy='max_matching')
        eval_results = metric.compute_metrics(fake_results)
        target_result = {'precision': 1, 'recall': 1, 'hmean': 1}
        self.assertDictEqual(target_result, eval_results)

# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np
import torch
from mmengine import InstanceData
from parameterized import parameterized

from mmocr.core import TextDetDataSample
from mmocr.models.textdet.postprocessors import FCEPostprocessor


class TestFCEPostProcessor(unittest.TestCase):

    def test_split_results(self):
        pred_results = [
            dict(
                cls_res=torch.rand(2, 4, 10, 10),
                reg_res=torch.rand(2, 21, 10, 10)),
            dict(
                cls_res=torch.rand(2, 4, 20, 20),
                reg_res=torch.rand(2, 21, 20, 20)),
            dict(
                cls_res=torch.rand(2, 4, 40, 40),
                reg_res=torch.rand(2, 21, 40, 40)),
        ]
        postprocessor = FCEPostprocessor(
            fourier_degree=5, num_reconstr_points=20, score_thr=0.3)
        split_results = postprocessor.split_results(pred_results)
        self.assertEqual(len(split_results), 2)
        self.assertEqual(len(split_results[0]), 3)
        self.assertEqual(len(split_results[1]), 3)
        self.assertEqual(split_results[0][0]['cls_res'].shape, (4, 10, 10))
        self.assertEqual(split_results[0][0]['reg_res'].shape, (21, 10, 10))

    @parameterized.expand([('poly'), ('quad')])
    def test_get_text_instances(self, text_repr_type):
        postprocessor = FCEPostprocessor(
            fourier_degree=5,
            num_reconstr_points=20,
            score_thr=0.3,
            text_repr_type=text_repr_type)
        pred_result = [
            dict(
                cls_res=torch.rand(4, 10, 10), reg_res=torch.rand(22, 10, 10)),
            dict(
                cls_res=torch.rand(4, 20, 20), reg_res=torch.rand(22, 20, 20)),
            dict(
                cls_res=torch.rand(4, 30, 30), reg_res=torch.rand(22, 30, 30)),
        ]
        data_sample = TextDetDataSample(
            gt_instances=InstanceData(polygons=[
                np.array([0, 0, 0, 1, 2, 1, 2, 0]),
                np.array([1, 1, 1, 2, 3, 2, 3, 1])
            ]))
        results = postprocessor.get_text_instances(pred_result, data_sample)
        self.assertIn('polygons', results.pred_instances)
        self.assertIn('scores', results.pred_instances)
        self.assertTrue(
            isinstance(results.pred_instances.scores, torch.FloatTensor))
        self.assertEqual(
            len(results.pred_instances.scores),
            len(results.pred_instances.polygons))
        if len(results.pred_instances.polygons) > 0:
            if text_repr_type == 'poly':
                self.assertEqual(results.pred_instances.polygons[0].shape,
                                 (40, ))
            else:
                self.assertEqual(results.pred_instances.polygons[0].shape,
                                 (8, ))

# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np
import torch
from mmengine import InstanceData
from nose_parameterized import parameterized

from mmocr.core import TextDetDataSample
from mmocr.models.textdet.postprocessors import DBPostprocessor


class TestDBPostProcessor(unittest.TestCase):

    def test_get_bbox_score(self):
        postprocessor = DBPostprocessor()
        score_map = np.arange(0, 1, step=0.05).reshape(4, 5)
        poly_pts = np.array(((0, 0), (0, 1), (1, 1), (1, 0)))
        self.assertAlmostEqual(
            postprocessor._get_bbox_score(score_map, poly_pts), 0.15)

    @parameterized.expand([('poly'), ('quad')])
    def test_get_text_instances(self, text_repr_type):

        postprocessor = DBPostprocessor(text_repr_type=text_repr_type)
        pred_result = dict(prob_map=torch.rand(4, 5))
        data_sample = TextDetDataSample(
            metainfo=dict(scale_factor=(0.5, 1)),
            gt_instances=InstanceData(polygons=[
                np.array([0, 0, 0, 1, 2, 1, 2, 0]),
                np.array([1, 1, 1, 2, 3, 2, 3, 1])
            ]))
        results = postprocessor.get_text_instances(pred_result, data_sample)
        self.assertIn('polygons', results.pred_instances)
        self.assertIn('scores', results.pred_instances)
        self.assertTrue(
            isinstance(results.pred_instances['scores'], torch.FloatTensor))

        postprocessor = DBPostprocessor(
            min_text_score=1, text_repr_type=text_repr_type)
        pred_result = dict(prob_map=torch.rand(4, 5) * 0.8)
        results = postprocessor.get_text_instances(pred_result, data_sample)
        self.assertEqual(results.pred_instances.polygons, [])
        self.assertTrue(
            isinstance(results.pred_instances['scores'], torch.FloatTensor))
        self.assertEqual(len(results.pred_instances.scores), 0)

# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np
import torch
from nose_parameterized import parameterized

from mmocr.core import TextDetDataSample
from mmocr.core.evaluation.utils import points2polygon, poly_iou
from mmocr.models.textdet.postprocessors import PANPostprocessor


class TestPANPostprocessor(unittest.TestCase):

    @parameterized.expand([('poly'), ('quad')])
    def test_get_text_instances(self, text_repr_type):
        postprocessor = PANPostprocessor(text_repr_type=text_repr_type)
        pred_result = torch.rand(6, 4, 5)
        data_sample = TextDetDataSample(metainfo=dict(scale_factor=(0.5, 1)))
        results = postprocessor.get_text_instances(pred_result, data_sample)
        self.assertIn('polygons', results.pred_instances)
        self.assertIn('scores', results.pred_instances)

        postprocessor = PANPostprocessor(
            min_text_confidence=1, text_repr_type=text_repr_type)
        pred_result = torch.rand(6, 4, 5) * 0.8
        results = postprocessor.get_text_instances(pred_result, data_sample)
        self.assertEqual(results.pred_instances.polygons, [])
        self.assertTrue(
            (results.pred_instances.scores == torch.FloatTensor([])).all())

    def test_points2boundary(self):

        postprocessor = PANPostprocessor(text_repr_type='quad')

        # test invalid arguments
        with self.assertRaises(AssertionError):
            postprocessor._points2boundary([])

        points = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1],
                           [0, 2], [1, 2], [2, 2]])

        # test quad
        postprocessor = PANPostprocessor(text_repr_type='quad')

        result = postprocessor._points2boundary(points)
        pred_poly = points2polygon(result)
        target_poly = points2polygon([2, 2, 0, 2, 0, 0, 2, 0])
        self.assertEqual(poly_iou(pred_poly, target_poly), 1)

        result = postprocessor._points2boundary(points, min_width=3)
        self.assertEqual(len(result), 0)

        # test poly
        postprocessor = PANPostprocessor(text_repr_type='poly')
        result = postprocessor._points2boundary(points)
        pred_poly = points2polygon(result)
        target_poly = points2polygon([0, 0, 0, 2, 2, 2, 2, 0])
        assert poly_iou(pred_poly, target_poly) == 1

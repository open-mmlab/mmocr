# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch
from parameterized import parameterized

from mmocr.core import TextDetDataSample
from mmocr.models.textdet.postprocessors import PSEPostprocessor


class TestPSEPostprocessor(unittest.TestCase):

    @parameterized.expand([('poly'), ('quad')])
    def test_get_text_instances(self, text_repr_type):
        postprocessor = PSEPostprocessor(text_repr_type=text_repr_type)
        pred_result = torch.rand(6, 4, 5)
        data_sample = TextDetDataSample(metainfo=dict(scale_factor=(0.5, 1)))
        results = postprocessor.get_text_instances(pred_result, data_sample)
        self.assertIn('polygons', results.pred_instances)
        self.assertIn('scores', results.pred_instances)

        postprocessor = PSEPostprocessor(
            score_threshold=1,
            min_kernel_confidence=1,
            text_repr_type=text_repr_type)
        pred_result = torch.rand(6, 4, 5) * 0.8
        results = postprocessor.get_text_instances(pred_result, data_sample)
        self.assertEqual(results.pred_instances.polygons, [])
        self.assertTrue(
            (results.pred_instances.scores == torch.FloatTensor([])).all())

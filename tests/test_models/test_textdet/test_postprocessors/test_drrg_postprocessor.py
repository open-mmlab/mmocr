# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np
import torch
from mmengine import InstanceData

from mmocr.core import TextDetDataSample
from mmocr.models.textdet.postprocessors import DRRGPostprocessor


class TestDRRGPostProcessor(unittest.TestCase):

    def test_call(self):

        postprocessor = DRRGPostprocessor()
        pred_results = (np.random.randint(0, 2, (10, 2)), np.random.rand(10),
                        np.random.rand(2, 9))
        data_sample = TextDetDataSample(
            metainfo=dict(scale_factor=(0.5, 1)),
            gt_instances=InstanceData(polygons=[
                np.array([0, 0, 0, 1, 2, 1, 2, 0]),
                np.array([1, 1, 1, 2, 3, 2, 3, 1])
            ]))
        result = postprocessor(pred_results, [data_sample])[0]
        self.assertIn('polygons', result.pred_instances)
        self.assertIn('scores', result.pred_instances)
        self.assertTrue(
            isinstance(result.pred_instances['scores'], torch.FloatTensor))

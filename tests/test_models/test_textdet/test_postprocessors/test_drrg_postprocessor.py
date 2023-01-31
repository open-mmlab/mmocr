# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np
import torch

from mmengine.structures import InstanceData
from mmocr.models.textdet.postprocessors import DRRGPostprocessor
from mmocr.structures import TextDetDataSample


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

    def test_comps2polys(self):
        postprocessor = DRRGPostprocessor()

        x1 = np.arange(2, 18, 2)
        x2 = x1 + 2
        y1 = np.ones(8) * 2
        y2 = y1 + 2
        comp_scores = np.ones(8, dtype=np.float32) * 0.9
        text_comps = np.stack([x1, y1, x2, y1, x2, y2, x1, y2,
                               comp_scores]).transpose()
        comp_labels = np.array([1, 1, 1, 1, 1, 3, 5, 5])
        shuffle = [3, 2, 5, 7, 6, 0, 4, 1]

        boundaries = postprocessor._comps2polys(text_comps[shuffle],
                                                comp_labels[shuffle])
        self.assertEqual(len(boundaries[0]), 3)

        boundaries = postprocessor._comps2polys(text_comps[[]],
                                                comp_labels[[]])
        self.assertEqual(len(boundaries[0]), 0)

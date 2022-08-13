# Copyright (c) OpenMMLab. All rights reserved.
import copy
import unittest

import numpy as np
import torch
from mmengine import InstanceData
from parameterized import parameterized

from mmocr.models.textdet.postprocessors import TextSnakePostprocessor
from mmocr.structures import TextDetDataSample


class TestTextSnakePostProcessor(unittest.TestCase):

    def setUp(self):
        # test decoding with text center region of small area
        maps = torch.zeros((1, 5, 224, 224), dtype=torch.float)
        maps[:, 0:2, :, :] = -10.
        maps[:, 0, 60:100, 50:170] = 10.
        maps[:, 1, 75:85, 60:160] = 10.
        maps[:, 2, 75:85, 60:160] = 0.
        maps[:, 3, 75:85, 60:160] = 1.
        maps[:, 4, 75:85, 60:160] = 10.
        maps[:, 0:2, 150:152, 5:7] = 10.
        self.pred_result1 = copy.deepcopy(maps)
        # test decoding with small radius
        maps.fill_(0.)
        maps[:, 0:2, :, :] = -10.
        maps[:, 0, 120:140, 20:40] = 10.
        maps[:, 1, 120:140, 20:40] = 10.
        maps[:, 2, 120:140, 20:40] = 0.
        maps[:, 3, 120:140, 20:40] = 1.
        maps[:, 4, 120:140, 20:40] = 0.5
        self.pred_result2 = copy.deepcopy(maps)

        self.data_sample = TextDetDataSample(
            metainfo=dict(scale_factor=(0.5, 1)),
            gt_instances=InstanceData(polygons=[
                np.array([0, 0, 0, 1, 2, 1, 2, 0]),
                np.array([1, 1, 1, 2, 3, 2, 3, 1])
            ]))

    @parameterized.expand([('poly')])
    def test_get_text_instances(self, text_repr_type):
        postprocessor = TextSnakePostprocessor(text_repr_type=text_repr_type)

        results = postprocessor.get_text_instances(
            torch.squeeze(self.pred_result1), self.data_sample)
        self.assertEqual(len(results.pred_instances.polygons), 1)

        results = postprocessor.get_text_instances(
            torch.squeeze(self.pred_result2), self.data_sample)
        self.assertEqual(len(results.pred_instances.polygons), 0)

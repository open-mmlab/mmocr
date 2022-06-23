# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import mock

import numpy as np
from mmengine import InstanceData

from mmocr.core import TextDetDataSample
from mmocr.models.textdet.postprocessors import BaseTextDetPostProcessor


class TestBaseTextDetPostProcessor(unittest.TestCase):

    def test_init(self):
        with self.assertRaises(AssertionError):
            BaseTextDetPostProcessor(text_repr_type='polygon')
        with self.assertRaises(AssertionError):
            BaseTextDetPostProcessor(rescale_fields='bbox')
        with self.assertRaises(AssertionError):
            BaseTextDetPostProcessor(train_cfg='test')
        with self.assertRaises(AssertionError):
            BaseTextDetPostProcessor(test_cfg='test')

    @mock.patch(f'{__name__}.BaseTextDetPostProcessor.get_text_instances')
    def test_call(self, mock_get_text_instances):

        def mock_func(x, y, **kwargs):
            return y

        mock_get_text_instances.side_effect = mock_func

        pred_results = {
            'prob_map': np.array([[0.1, 0.2], [0.3, 0.4]]),
        }
        data_samples = [
            TextDetDataSample(
                metainfo=dict(scale_factor=(0.5, 1)),
                pred_instances=InstanceData(
                    polygons=[np.array([0, 0, 0, 1, 2, 1, 2, 0])])),
            TextDetDataSample(
                metainfo=dict(scale_factor=(1, 0.5)),
                pred_instances=InstanceData(polygons=[
                    np.array([0, 0, 0, 1, 2, 1, 2, 0]),
                    np.array([1, 1, 1, 2, 3, 2, 3, 1])
                ]))
        ]
        base_postprocessor = BaseTextDetPostProcessor(
            rescale_fields=['polygons'])
        results = base_postprocessor(pred_results, data_samples)
        self.assertEqual(len(results), 2)
        self.assertTrue(
            np.array_equal(results[0].pred_instances.polygons,
                           [np.array([0, 0, 0, 1, 4, 1, 4, 0])]))
        self.assertTrue(
            np.array_equal(results[1].pred_instances.polygons, [
                np.array([0, 0, 0, 2, 2, 2, 2, 0]),
                np.array([1, 2, 1, 4, 3, 4, 3, 2])
            ]))

    def test_rescale(self):

        data_sample = TextDetDataSample()
        data_sample.pred_instances = InstanceData()
        data_sample.pred_instances.polygons = [
            np.array([0, 0, 0, 1, 1, 1, 1, 0])
        ]

        base_postprocessor = BaseTextDetPostProcessor(
            text_repr_type='poly', rescale_fields=['polygons'])
        rescaled_data_sample = base_postprocessor.rescale(
            data_sample, (0.5, 1))
        self.assertTrue(
            np.array_equal(rescaled_data_sample.pred_instances.polygons,
                           [[0, 0, 0, 1, 2, 1, 2, 0]]))

    def test_get_text_instances(self):
        with self.assertRaises(NotImplementedError):
            BaseTextDetPostProcessor().get_text_instances(None, None)

    def test_split_results(self):

        base_postprocessor = BaseTextDetPostProcessor()

        # test invalid arguments
        with self.assertRaises(AssertionError):
            base_postprocessor.split_results(None)

        with self.assertRaises(AssertionError):
            base_postprocessor.split_results({'test': [0, 1]}, 'fields')

        with self.assertRaises(AssertionError):
            base_postprocessor.split_results({'test': [0, 1]},
                                             keep_unsplit_fields='true')

        # test split_results
        results = {
            'test': [0, 1],
            'test2': np.array([2, 3], dtype=int),
            'meta': 'str'
        }
        split_results = base_postprocessor.split_results(results, ['test'])
        self.assertEqual(split_results, [{'test': 0}, {'test': 1}])

        split_results = base_postprocessor.split_results(
            results, ['test', 'test2'])
        self.assertEqual(split_results, [{
            'test': 0,
            'test2': 2
        }, {
            'test': 1,
            'test2': 3
        }])

        split_results = base_postprocessor.split_results(
            results, ['test', 'test2'], keep_unsplit_fields=True)
        self.assertEqual(split_results, [{
            'test': 0,
            'test2': 2,
            'meta': 'str'
        }, {
            'test': 1,
            'test2': 3,
            'meta': 'str'
        }])

    def test_poly_nms(self):
        base_postprocessor = BaseTextDetPostProcessor(text_repr_type='poly')
        polygons = [
            np.array([0., 0., 10., 0., 10., 10., 0., 10.]),
            np.array([5., 0., 15., 0., 15., 10., 5., 10.])
        ]
        scores = [0.9, 0.8]
        keep = base_postprocessor.poly_nms(polygons, scores, 0.6)

        self.assertEqual(len(keep[0]), 2)
        self.assertTrue(np.allclose(keep[0][0], polygons[0]))
        self.assertTrue(np.allclose(keep[0][1], polygons[1]))

        keep = base_postprocessor.poly_nms(polygons, scores, 0.2)
        self.assertEqual(len(keep[0]), 1)
        self.assertTrue(np.allclose(keep[0][0], polygons[0]))

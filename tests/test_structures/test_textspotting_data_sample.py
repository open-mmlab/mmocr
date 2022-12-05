# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch
from mmengine.structures import InstanceData

from mmocr.structures import TextSpottingDataSample


class TestTextSpottingDataSample(TestCase):

    def _equal(self, a, b):
        if isinstance(a, (torch.Tensor, np.ndarray)):
            return (a == b).all()
        else:
            return a == b

    def test_init(self):
        meta_info = dict(
            img_size=[256, 256],
            scale_factor=np.array([1.5, 1.5]),
            img_shape=torch.rand(4))

        det_data_sample = TextSpottingDataSample(metainfo=meta_info)
        assert 'img_size' in det_data_sample

        self.assertListEqual(det_data_sample.img_size, [256, 256])
        self.assertListEqual(det_data_sample.get('img_size'), [256, 256])

    def test_setter(self):
        det_data_sample = TextSpottingDataSample()
        # test gt_instances
        gt_instances_data = dict(
            bboxes=torch.rand(4, 4),
            labels=torch.rand(4),
            masks=np.random.rand(4, 2, 2))
        gt_instances = InstanceData(**gt_instances_data)
        det_data_sample.gt_instances = gt_instances
        assert 'gt_instances' in det_data_sample
        assert self._equal(det_data_sample.gt_instances.bboxes,
                           gt_instances_data['bboxes'])
        assert self._equal(det_data_sample.gt_instances.labels,
                           gt_instances_data['labels'])
        assert self._equal(det_data_sample.gt_instances.masks,
                           gt_instances_data['masks'])

        # test pred_instances
        pred_instances_data = dict(
            bboxes=torch.rand(2, 4),
            labels=torch.rand(2),
            masks=np.random.rand(2, 2, 2))
        pred_instances = InstanceData(**pred_instances_data)
        det_data_sample.pred_instances = pred_instances
        assert 'pred_instances' in det_data_sample
        assert self._equal(det_data_sample.pred_instances.bboxes,
                           pred_instances_data['bboxes'])
        assert self._equal(det_data_sample.pred_instances.labels,
                           pred_instances_data['labels'])
        assert self._equal(det_data_sample.pred_instances.masks,
                           pred_instances_data['masks'])

        # test type error
        with self.assertRaises(AssertionError):
            det_data_sample.gt_instances = torch.rand(2, 4)
        with self.assertRaises(AssertionError):
            det_data_sample.pred_instances = torch.rand(2, 4)

    def test_deleter(self):
        gt_instances_data = dict(
            bboxes=torch.rand(4, 4),
            labels=torch.rand(4),
            masks=np.random.rand(4, 2, 2))

        det_data_sample = TextSpottingDataSample()
        gt_instances = InstanceData(data=gt_instances_data)
        det_data_sample.gt_instances = gt_instances
        assert 'gt_instances' in det_data_sample
        del det_data_sample.gt_instances
        assert 'gt_instances' not in det_data_sample

        det_data_sample.pred_instances = gt_instances
        assert 'pred_instances' in det_data_sample
        del det_data_sample.pred_instances
        assert 'pred_instances' not in det_data_sample

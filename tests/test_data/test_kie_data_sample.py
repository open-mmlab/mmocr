# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch
from mmengine.data import InstanceData

from mmocr.data import KIEDataSample


class TestTextDetDataSample(TestCase):

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

        kie_data_sample = KIEDataSample(metainfo=meta_info)
        assert 'img_size' in kie_data_sample

        self.assertListEqual(kie_data_sample.img_size, [256, 256])
        self.assertListEqual(kie_data_sample.get('img_size'), [256, 256])

    def test_setter(self):
        kie_data_sample = KIEDataSample()
        # test gt_instances
        gt_instances_data = dict(
            bboxes=torch.rand(4, 4),
            labels=torch.rand(4),
            texts=['t1', 't2', 't3', 't4'],
            relations=torch.rand(4, 4),
            edge_labels=torch.randint(0, 4, (4, )))
        gt_instances = InstanceData(**gt_instances_data)
        kie_data_sample.gt_instances = gt_instances
        self.assertIn('gt_instances', kie_data_sample)
        self.assertTrue(
            self._equal(kie_data_sample.gt_instances.bboxes,
                        gt_instances_data['bboxes']))
        self.assertTrue(
            self._equal(kie_data_sample.gt_instances.labels,
                        gt_instances_data['labels']))
        self.assertTrue(
            self._equal(kie_data_sample.gt_instances.texts,
                        gt_instances_data['texts']))
        self.assertTrue(
            self._equal(kie_data_sample.gt_instances.relations,
                        gt_instances_data['relations']))
        self.assertTrue(
            self._equal(kie_data_sample.gt_instances.edge_labels,
                        gt_instances_data['edge_labels']))

        # test pred_instances
        pred_instances_data = dict(
            bboxes=torch.rand(4, 4),
            labels=torch.rand(4),
            texts=['t1', 't2', 't3', 't4'],
            relations=torch.rand(4, 4),
            edge_labels=torch.randint(0, 4, (4, )))
        pred_instances = InstanceData(**pred_instances_data)
        kie_data_sample.pred_instances = pred_instances
        assert 'pred_instances' in kie_data_sample
        assert self._equal(kie_data_sample.pred_instances.bboxes,
                           pred_instances_data['bboxes'])
        assert self._equal(kie_data_sample.pred_instances.labels,
                           pred_instances_data['labels'])
        self.assertTrue(
            self._equal(kie_data_sample.pred_instances.texts,
                        pred_instances_data['texts']))
        self.assertTrue(
            self._equal(kie_data_sample.pred_instances.relations,
                        pred_instances_data['relations']))
        self.assertTrue(
            self._equal(kie_data_sample.pred_instances.edge_labels,
                        pred_instances_data['edge_labels']))

        # test type error
        with self.assertRaises(AssertionError):
            kie_data_sample.gt_instances = torch.rand(2, 4)
        with self.assertRaises(AssertionError):
            kie_data_sample.pred_instances = torch.rand(2, 4)

    def test_deleter(self):
        gt_instances_data = dict(
            bboxes=torch.rand(4, 4),
            labels=torch.rand(4),
        )

        kie_data_sample = KIEDataSample()
        gt_instances = InstanceData(data=gt_instances_data)
        kie_data_sample.gt_instances = gt_instances
        assert 'gt_instances' in kie_data_sample
        del kie_data_sample.gt_instances
        assert 'gt_instances' not in kie_data_sample

        kie_data_sample.pred_instances = gt_instances
        assert 'pred_instances' in kie_data_sample
        del kie_data_sample.pred_instances
        assert 'pred_instances' not in kie_data_sample

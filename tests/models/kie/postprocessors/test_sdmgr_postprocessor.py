# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest import TestCase

import torch
from mmengine import InstanceData

from mmocr.models.kie.postprocessors import SDMGRPostProcessor
from mmocr.structures import KIEDataSample


class TestSDMGRPostProcessor(TestCase):

    def setUp(self):
        node_preds = self.rand_prob_dist(6, 3)
        edge_preds = self.rand_prob_dist(20, 2)
        self.preds = (node_preds, edge_preds)

        data_sample1 = KIEDataSample()
        data_sample1.gt_instances = InstanceData(
            bboxes=torch.randint(0, 26, (2, 4)).long())
        data_sample2 = KIEDataSample()
        data_sample2.gt_instances = InstanceData(
            bboxes=torch.randint(0, 26, (4, 4)).long())
        self.data_samples = [data_sample1, data_sample2]

    def rand_prob_dist(self, batch_num: int, n_classes: int) -> torch.Tensor:
        assert n_classes > 1
        result = torch.zeros((batch_num, n_classes))
        result[:, 0] = torch.rand((batch_num, ))
        diff = 1 - result[:, 0]
        for i in range(1, n_classes - 1):
            result[:, i] = diff * torch.rand((batch_num, ))
            diff -= result[:, i]
        result[:, -1] = diff
        return result

    def test_init(self):
        with self.assertRaises(AssertionError):
            SDMGRPostProcessor(link_type=1)

        with self.assertRaises(AssertionError):
            SDMGRPostProcessor(link_type='one-to-one')

    def test_forward(self):
        postprocessor = SDMGRPostProcessor()
        data_samples = postprocessor(self.preds,
                                     copy.deepcopy(self.data_samples))
        self.assertEqual(data_samples[0].pred_instances.labels.shape, (2, ))
        self.assertEqual(data_samples[0].pred_instances.scores.shape, (2, ))
        self.assertEqual(data_samples[0].pred_instances.edge_labels.shape,
                         (2, 2))
        self.assertEqual(data_samples[0].pred_instances.edge_scores.shape,
                         (2, 2))
        self.assertEqual(data_samples[1].pred_instances.labels.shape, (4, ))
        self.assertEqual(data_samples[1].pred_instances.scores.shape, (4, ))
        self.assertEqual(data_samples[1].pred_instances.edge_labels.shape,
                         (4, 4))
        self.assertEqual(data_samples[1].pred_instances.edge_scores.shape,
                         (4, 4))

    def test_one_to_one(self):
        postprocessor = SDMGRPostProcessor(
            link_type='one-to-one', key_node_idx=1, value_node_idx=2)
        data_samples = postprocessor(self.preds,
                                     copy.deepcopy(self.data_samples))
        for data_sample in data_samples:
            tails, heads = torch.where(
                data_sample.pred_instances.edge_labels == 1)
            if len(tails) > 0:
                self.assertTrue(
                    (data_sample.pred_instances.labels[tails] == 1).all())
                self.assertEqual(len(set(tails.numpy().tolist())), len(tails))
            if len(heads) > 0:
                self.assertTrue(
                    (data_sample.pred_instances.labels[heads] == 2).all())
                self.assertEqual(len(set(heads.numpy().tolist())), len(heads))

    def test_one_to_many(self):
        postprocessor = SDMGRPostProcessor(
            link_type='one-to-many', key_node_idx=1, value_node_idx=2)
        data_samples = postprocessor(self.preds,
                                     copy.deepcopy(self.data_samples))
        for data_sample in data_samples:
            tails, heads = torch.where(
                data_sample.pred_instances.edge_labels == 1)
            if len(tails) > 0:
                self.assertTrue(
                    (data_sample.pred_instances.labels[tails] == 1).all())
            if len(heads) > 0:
                self.assertTrue(
                    (data_sample.pred_instances.labels[heads] == 2).all())
                self.assertEqual(len(set(heads.numpy().tolist())), len(heads))

    def test_many_to_many(self):
        postprocessor = SDMGRPostProcessor(
            link_type='many-to-many', key_node_idx=1, value_node_idx=2)
        data_samples = postprocessor(self.preds,
                                     copy.deepcopy(self.data_samples))
        for data_sample in data_samples:
            tails, heads = torch.where(
                data_sample.pred_instances.edge_labels == 1)
            if len(tails) > 0:
                self.assertTrue(
                    (data_sample.pred_instances.labels[tails] == 1).all())
            if len(heads) > 0:
                self.assertTrue(
                    (data_sample.pred_instances.labels[heads] == 2).all())

    def test_many_to_one(self):
        postprocessor = SDMGRPostProcessor(
            link_type='many-to-one', key_node_idx=1, value_node_idx=2)
        data_samples = postprocessor(self.preds,
                                     copy.deepcopy(self.data_samples))
        for data_sample in data_samples:
            tails, heads = torch.where(
                data_sample.pred_instances.edge_labels == 1)
            if len(tails) > 0:
                self.assertTrue(
                    (data_sample.pred_instances.labels[tails] == 1).all())
                self.assertEqual(len(set(tails.numpy().tolist())), len(tails))
            if len(heads) > 0:
                self.assertTrue(
                    (data_sample.pred_instances.labels[heads] == 2).all())

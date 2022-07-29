# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine import InstanceData

from mmocr.data import KIEDataSample
from mmocr.models.kie.module_losses import SDMGRModuleLoss


class TestSDMGRModuleLoss(TestCase):

    def test_forward(self):
        loss = SDMGRModuleLoss()

        node_preds = torch.rand((3, 26))
        edge_preds = torch.rand((9, 2))
        data_sample = KIEDataSample()
        data_sample.gt_instances = InstanceData(
            labels=torch.randint(0, 26, (3, )).long(),
            edge_labels=torch.randint(0, 2, (3, 3)).long())

        losses = loss((node_preds, edge_preds), [data_sample])
        self.assertIn('loss_node', losses)
        self.assertIn('loss_edge', losses)
        self.assertIn('acc_node', losses)
        self.assertIn('acc_edge', losses)

        loss = SDMGRModuleLoss(weight_edge=2, weight_node=3)
        new_losses = loss((node_preds, edge_preds), [data_sample])
        self.assertEqual(losses['loss_node'] * 3, new_losses['loss_node'])
        self.assertEqual(losses['loss_edge'] * 2, new_losses['loss_edge'])

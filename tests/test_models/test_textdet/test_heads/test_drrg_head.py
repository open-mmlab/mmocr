# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase, mock

import numpy as np
import torch

from mmocr.models.textdet.heads.drrg_head import (GCN, DRRGHead, LocalGraphs,
                                                  ProposalLocalGraphs,
                                                  feature_embedding,
                                                  normalize_adjacent_matrix)


class TestDRRGHead(TestCase):

    def setUp(self) -> None:
        self.drrg_head = DRRGHead(in_channels=10)

    @mock.patch('mmocr.models.textdet.losses.drrg_loss.DRRGLoss.get_targets')
    @mock.patch('mmocr.models.textdet.losses.drrg_loss.DRRGLoss.forward')
    def test_loss(self, mock_forward, mock_get_targets):
        num_rois = 16
        feature_maps = torch.randn((2, 10, 128, 128), dtype=torch.float)
        x = np.random.randint(4, 124, (num_rois, 1))
        y = np.random.randint(4, 124, (num_rois, 1))
        h = 4 * np.ones((num_rois, 1))
        w = 4 * np.ones((num_rois, 1))
        angle = (np.random.random_sample((num_rois, 1)) * 2 - 1) * np.pi / 2
        cos, sin = np.cos(angle), np.sin(angle)
        comp_labels = np.random.randint(1, 3, (num_rois, 1))
        num_rois = num_rois * np.ones((num_rois, 1))
        comp_attribs = np.hstack([num_rois, x, y, h, w, cos, sin, comp_labels])
        comp_attribs = comp_attribs.astype(np.float32)
        comp_attribs_ = comp_attribs.copy()
        comp_attribs = np.stack([comp_attribs, comp_attribs_])
        mock_get_targets.return_value = (None, None, None, None, None, None,
                                         None, comp_attribs)
        mock_forward.side_effect = lambda *args: args[0]
        # It returns the tensor input to module loss
        pred_maps, pred_labels, gt_labels = self.drrg_head.loss(
            feature_maps, None)
        self.assertEqual(pred_maps.size(), (2, 6, 128, 128))
        self.assertTrue(pred_labels.ndim == gt_labels.ndim == 2)
        self.assertEqual(gt_labels.size()[0] * gt_labels.size()[1],
                         pred_labels.size()[0])
        self.assertEqual(pred_labels.size()[1], 2)

    def test_predict(self):
        with torch.no_grad():
            feat_maps = torch.zeros((1, 10, 128, 128))
            self.drrg_head.out_conv.bias.data.fill_(-10)
            preds = self.drrg_head(feat_maps)
            self.assertTrue(all([pred is None for pred in preds]))


class TestLocalGraphs(TestCase):

    def test_call(self):
        geo_feat_len = 24
        pooling_h, pooling_w = pooling_out_size = (2, 2)
        num_rois = 32

        local_graph_generator = LocalGraphs((4, 4), 3, geo_feat_len, 1.0,
                                            pooling_out_size, 0.5)

        feature_maps = torch.randn((2, 3, 128, 128), dtype=torch.float)
        x = np.random.randint(4, 124, (num_rois, 1))
        y = np.random.randint(4, 124, (num_rois, 1))
        h = 4 * np.ones((num_rois, 1))
        w = 4 * np.ones((num_rois, 1))
        angle = (np.random.random_sample((num_rois, 1)) * 2 - 1) * np.pi / 2
        cos, sin = np.cos(angle), np.sin(angle)
        comp_labels = np.random.randint(1, 3, (num_rois, 1))
        num_rois = num_rois * np.ones((num_rois, 1))
        comp_attribs = np.hstack([num_rois, x, y, h, w, cos, sin, comp_labels])
        comp_attribs = comp_attribs.astype(np.float32)
        comp_attribs_ = comp_attribs.copy()
        comp_attribs = np.stack([comp_attribs, comp_attribs_])

        (node_feats, adjacent_matrix, knn_inds,
         linkage_labels) = local_graph_generator(feature_maps, comp_attribs)
        feat_len = geo_feat_len + \
            feature_maps.size()[1] * pooling_h * pooling_w

        self.assertTrue(node_feats.dim() == adjacent_matrix.dim() == 3)
        self.assertEqual(node_feats.size()[-1], feat_len)
        self.assertEqual(knn_inds.size()[-1], 4)
        self.assertEqual(linkage_labels.size()[-1], 4)
        self.assertTrue(node_feats.size()[0] == adjacent_matrix.size()[0] ==
                        knn_inds.size()[0] == linkage_labels.size()[0])
        self.assertTrue(node_feats.size()[1] == adjacent_matrix.size()[1] ==
                        adjacent_matrix.size()[2])


class TestProposalLocalGraphs(TestCase):

    def test_call(self):
        geo_feat_len = 24
        pooling_h, pooling_w = pooling_out_size = (2, 2)

        local_graph_generator = ProposalLocalGraphs(
            (4, 4), 2, geo_feat_len, 1., pooling_out_size, 0.1, 3., 6., 1.,
            0.5, 0.3, 0.5, 0.5, 2)

        maps = torch.zeros((1, 6, 224, 224), dtype=torch.float)
        maps[:, 0:2, :, :] = -10.
        maps[:, 0, 60:100, 50:170] = 10.
        maps[:, 1, 75:85, 60:160] = 10.
        maps[:, 2, 75:85, 60:160] = 0.
        maps[:, 3, 75:85, 60:160] = 1.
        maps[:, 4, 75:85, 60:160] = 10.
        maps[:, 5, 75:85, 60:160] = 10.
        feature_maps = torch.randn((2, 6, 224, 224), dtype=torch.float)
        feat_len = geo_feat_len + \
            feature_maps.size()[1] * pooling_h * pooling_w

        none_flag, graph_data = local_graph_generator(maps, feature_maps)
        (node_feats, adjacent_matrices, knn_inds, local_graphs,
         text_comps) = graph_data

        self.assertFalse(none_flag, False)
        self.assertEqual(text_comps.ndim, 2)
        self.assertGreater(text_comps.shape[0], 0)
        self.assertEqual(text_comps.shape[1], 9)
        self.assertTrue(
            node_feats.size()[0] == adjacent_matrices.size()[0] == knn_inds.
            size()[0] == local_graphs.size()[0] == text_comps.shape[0])
        self.assertTrue(node_feats.size()[1] == adjacent_matrices.size()[1] ==
                        adjacent_matrices.size()[2] == local_graphs.size()[1])
        self.assertEqual(node_feats.size()[-1], feat_len)

        # test proposal local graphs with area of center region less than
        # threshold
        maps[:, 1, 75:85, 60:160] = -10.
        maps[:, 1, 80, 80] = 10.
        none_flag, _ = local_graph_generator(maps, feature_maps)
        self.assertTrue(none_flag)

        # test proposal local graphs with one text component
        local_graph_generator = ProposalLocalGraphs(
            (4, 4), 2, geo_feat_len, 1., pooling_out_size, 0.1, 8., 20., 1.,
            0.5, 0.3, 0.5, 0.5, 2)
        maps[:, 1, 78:82, 78:82] = 10.
        none_flag, _ = local_graph_generator(maps, feature_maps)
        self.assertTrue(none_flag)

        # test proposal local graphs with text components out of text region
        maps[:, 0, 60:100, 50:170] = -10.
        maps[:, 0, 78:82, 78:82] = 10.
        none_flag, _ = local_graph_generator(maps, feature_maps)
        self.assertTrue(none_flag)


class TestUtils(TestCase):

    def test_normalize_adjacent_matrix(self):
        adjacent_matrix = np.random.randint(0, 2, (16, 16))
        normalized_matrix = normalize_adjacent_matrix(adjacent_matrix)
        self.assertEqual(normalized_matrix.shape, adjacent_matrix.shape)

    def test_feature_embedding(self):
        out_feat_len = 48

        # test without residue dimensions
        feats = np.random.randn(10, 8)
        embed_feats = feature_embedding(feats, out_feat_len)
        self.assertEqual(embed_feats.shape, (10, out_feat_len))

        # test with residue dimensions
        feats = np.random.randn(10, 9)
        embed_feats = feature_embedding(feats, out_feat_len)
        self.assertEqual(embed_feats.shape, (10, out_feat_len))


class TestGCN(TestCase):

    def test_forward(self):
        num_local_graphs = 32
        num_max_graph_nodes = 16
        input_feat_len = 512
        k = 8
        gcn = GCN(input_feat_len)
        node_feat = torch.randn(
            (num_local_graphs, num_max_graph_nodes, input_feat_len))
        adjacent_matrix = torch.rand(
            (num_local_graphs, num_max_graph_nodes, num_max_graph_nodes))
        knn_inds = torch.randint(1, num_max_graph_nodes, (num_local_graphs, k))
        output = gcn(node_feat, adjacent_matrix, knn_inds)
        self.assertEqual(output.size(), (num_local_graphs * k, 2))

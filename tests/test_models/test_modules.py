import numpy as np
import pytest
import torch

from mmocr.models.textdet.modules import GCN, LocalGraphs, ProposalLocalGraphs
from mmocr.models.textdet.modules.utils import (feature_embedding,
                                                normalize_adjacent_matrix)


def test_local_graph_forward_train():
    geo_feat_len = 24
    pooling_h, pooling_w = pooling_out_size = (2, 2)
    roi_num = 32

    local_graph_generator = LocalGraphs((4, 4), 3, geo_feat_len, 1.0,
                                        pooling_out_size, 0.5)

    feature_maps = torch.randn((2, 3, 128, 128), dtype=torch.float)
    x = np.random.randint(4, 124, (roi_num, 1))
    y = np.random.randint(4, 124, (roi_num, 1))
    h = 4 * np.ones((roi_num, 1))
    w = 4 * np.ones((roi_num, 1))
    angle = (np.random.random_sample((roi_num, 1)) * 2 - 1) * np.pi / 2
    cos, sin = np.cos(angle), np.sin(angle)
    comp_labels = np.random.randint(1, 3, (roi_num, 1))
    roi_num = roi_num * np.ones((roi_num, 1))
    comp_attribs = np.hstack([roi_num, x, y, h, w, cos, sin, comp_labels])
    comp_attribs = comp_attribs.astype(np.float32)
    comp_attribs_ = comp_attribs.copy()
    comp_attribs = np.stack([comp_attribs, comp_attribs_])

    (node_feats, adjacent_matrix, knn_inds,
     linkage_labels) = local_graph_generator(feature_maps, comp_attribs)
    feat_len = geo_feat_len + feature_maps.size()[1] * pooling_h * pooling_w

    assert node_feats.dim() == adjacent_matrix.dim() == 3
    assert node_feats.size()[-1] == feat_len
    assert knn_inds.size()[-1] == 4
    assert linkage_labels.size()[-1] == 4
    assert (node_feats.size()[0] == adjacent_matrix.size()[0] ==
            knn_inds.size()[0] == linkage_labels.size()[0])
    assert (node_feats.size()[1] == adjacent_matrix.size()[1] ==
            adjacent_matrix.size()[2])


def test_local_graph_forward_test():
    geo_feat_len = 24
    pooling_h, pooling_w = pooling_out_size = (2, 2)

    local_grpah_generator = ProposalLocalGraphs(
        (4, 4), 2, geo_feat_len, 1., pooling_out_size, 0.1, 3., 6., 1., 0.5,
        0.3, 0.5, 0.5, 2)

    maps = torch.zeros((1, 6, 224, 224), dtype=torch.float)
    maps[:, 0:2, :, :] = -10.
    maps[:, 0, 60:100, 12:212] = 10.
    maps[:, 1, 70:90, 22:202] = 10.
    maps[:, 2, 70:90, 22:202] = 0.
    maps[:, 3, 70:90, 22:202] = 1.
    maps[:, 4, 70:90, 22:202] = 5.
    maps[:, 5, 70:90, 22:202] = 5.
    feature_maps = torch.randn((2, 3, 128, 128), dtype=torch.float)
    feat_len = geo_feat_len + feature_maps.size()[1] * pooling_h * pooling_w

    none_flag, graph_data = local_grpah_generator(maps, feature_maps)
    (node_feats, adjacent_matrices, knn_inds, local_graphs,
     text_comps) = graph_data

    assert none_flag is False
    assert text_comps.ndim == 2
    assert text_comps.shape[0] > 0
    assert text_comps.shape[1] == 9
    assert (node_feats.size()[0] == adjacent_matrices.size()[0] ==
            knn_inds.size()[0] == local_graphs.size()[0] ==
            text_comps.shape[0])
    assert (node_feats.size()[1] == adjacent_matrices.size()[1] ==
            adjacent_matrices.size()[2] == local_graphs.size()[1])
    assert node_feats.size()[-1] == feat_len


def test_gcn():
    local_graph_num = 32
    max_graph_node_num = 16
    input_feat_len = 512
    k = 8
    gcn = GCN(input_feat_len)
    node_feat = torch.randn(
        (local_graph_num, max_graph_node_num, input_feat_len))
    adjacent_matrix = torch.rand(
        (local_graph_num, max_graph_node_num, max_graph_node_num))
    knn_inds = torch.randint(1, max_graph_node_num, (local_graph_num, k))
    output = gcn(node_feat, adjacent_matrix, knn_inds)
    assert output.size() == (local_graph_num * k, 2)


def test_normalize_adjacent_matrix():
    adjacent_matrix = np.random.randn(32, 32)
    normalized_matrix = normalize_adjacent_matrix(adjacent_matrix, mode='AD')
    assert normalized_matrix.shape == adjacent_matrix.shape

    normalized_matrix = normalize_adjacent_matrix(adjacent_matrix, mode='DAD')
    assert normalized_matrix.shape == adjacent_matrix.shape

    with pytest.raises(NotImplementedError):
        normalized_matrix = normalize_adjacent_matrix(
            adjacent_matrix, mode='DA')


def test_feature_embedding():
    out_feat_len = 48

    # test without residue dimensions
    feats = np.random.randn(10, 8)
    embed_feats = feature_embedding(feats, out_feat_len)
    assert embed_feats.shape == (10, out_feat_len)

    # test with residue dimensions
    feats = np.random.randn(10, 9)
    embed_feats = feature_embedding(feats, out_feat_len)
    assert embed_feats.shape == (10, out_feat_len)

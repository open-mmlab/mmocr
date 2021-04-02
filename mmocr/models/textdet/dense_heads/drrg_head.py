import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.models.builder import HEADS, build_loss
from mmocr.models.textdet.modules import (GCN, LocalGraphs,
                                          ProposalLocalGraphs,
                                          merge_text_comps)
from .head_mixin import HeadMixin


@HEADS.register_module()
class DRRGHead(HeadMixin, nn.Module):
    """The class for DRRG head: Deep Relational Reasoning Graph Network for
    Arbitrary Shape Text Detection.

    [https://arxiv.org/abs/2003.07493]

    Args:
        k_at_hops (tuple(int)): The number of i-hop neighbors,
            i = 1, 2, ..., h.
        active_connection (int): The number of two hop neighbors deem as
            linked to a pivot.
        node_geo_feat_dim (int): The dimension of embedded geometric features
            of a component.
        pooling_scale (float): The spatial scale of RRoI-Aligning.
        pooling_output_size (tuple(int)): The size of RRoI-Aligning output.
        graph_filter_thr (float): The threshold to filter identical local
            graphs.
        comp_shrink_ratio (float): The shrink ratio of text components.
        nms_thr (float): The  locality-aware NMS threshold.
        min_width (float): The minimum width of text components.
        max_width (float): The maximum width of text components.
        comp_ratio (float): The reciprocal of aspect ratio of text components.
        text_region_thr (float): The threshold for text region probability map.
        center_region_thr (float): The threshold of text center region
            probability map.
        center_region_area_thr (int): The threshold of filtering small-size
            text center region.
        link_thr (float): The threshold for connected components searching.
    """

    def __init__(self,
                 in_channels,
                 k_at_hops=(8, 4),
                 active_connection=3,
                 node_geo_feat_dim=120,
                 pooling_scale=1.0,
                 pooling_output_size=(3, 4),
                 graph_filter_thr=0.75,
                 comp_shrink_ratio=0.95,
                 nms_thr=0.25,
                 min_width=8.0,
                 max_width=24.0,
                 comp_ratio=0.65,
                 text_region_thr=0.6,
                 center_region_thr=0.4,
                 center_region_area_thr=100,
                 link_thr=0.85,
                 loss=dict(type='DRRGLoss'),
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()

        assert isinstance(in_channels, int)
        assert isinstance(k_at_hops, tuple)
        assert isinstance(active_connection, int)
        assert isinstance(node_geo_feat_dim, int)
        assert isinstance(pooling_scale, float)
        assert isinstance(pooling_output_size, tuple)
        assert isinstance(graph_filter_thr, float)
        assert isinstance(comp_shrink_ratio, float)
        assert isinstance(nms_thr, float)
        assert isinstance(min_width, float)
        assert isinstance(max_width, float)
        assert isinstance(comp_ratio, float)
        assert isinstance(center_region_area_thr, int)
        assert isinstance(link_thr, float)

        self.in_channels = in_channels
        self.out_channels = 6
        self.downsample_ratio = 1.0
        self.k_at_hops = k_at_hops
        self.active_connection = active_connection
        self.node_geo_feat_dim = node_geo_feat_dim
        self.pooling_scale = pooling_scale
        self.pooling_output_size = pooling_output_size
        self.graph_filter_thr = graph_filter_thr
        self.comp_shrink_ratio = comp_shrink_ratio
        self.nms_thr = nms_thr
        self.min_width = min_width
        self.max_width = max_width
        self.comp_ratio = comp_ratio
        self.text_region_thr = text_region_thr
        self.center_region_thr = center_region_thr
        self.center_region_area_thr = center_region_area_thr
        self.link_thr = link_thr
        self.loss_module = build_loss(loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.out_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0)
        self.init_weights()

        self.graph_train = LocalGraphs(self.k_at_hops, self.active_connection,
                                       self.node_geo_feat_dim,
                                       self.pooling_scale,
                                       self.pooling_output_size,
                                       self.graph_filter_thr)

        self.graph_test = ProposalLocalGraphs(
            self.k_at_hops, self.active_connection, self.node_geo_feat_dim,
            self.pooling_scale, self.pooling_output_size, self.nms_thr,
            self.min_width, self.max_width, self.comp_shrink_ratio,
            self.comp_ratio, self.text_region_thr, self.center_region_thr,
            self.center_region_area_thr)

        pool_w, pool_h = self.pooling_output_size
        gcn_in_dim = (pool_w * pool_h) * (
            self.in_channels + self.out_channels) + self.node_geo_feat_dim
        self.gcn = GCN(gcn_in_dim, 32)

    def init_weights(self):
        normal_init(self.out_conv, mean=0, std=0.01)

    def forward(self, inputs, text_comp_feats):

        pred_maps = self.out_conv(inputs)

        feat_maps = torch.cat([inputs, pred_maps], dim=1)
        node_feats, adjacent_matrices, knn_inx, gt_labels = self.graph_train(
            feat_maps, np.array(text_comp_feats))

        gcn_pred = self.gcn(node_feats, adjacent_matrices, knn_inx)

        return (pred_maps, (gcn_pred, gt_labels))

    def single_test(self, feat_maps):

        pred_maps = self.out_conv(feat_maps)
        feat_maps = torch.cat([feat_maps[0], pred_maps], dim=1)

        none_flag, graph_data = self.graph_test(pred_maps, feat_maps)

        (node_feats, adjacent_matrix, pivot_inx, knn_inx, local_graph_nodes,
         text_comps) = graph_data

        if none_flag:
            return None, None, None

        adjacent_matrix, pivot_inx, knn_inx = map(
            lambda x: x.to(feat_maps.device),
            (adjacent_matrix, pivot_inx, knn_inx))
        gcn_pred = self.gcn_model(node_feats, adjacent_matrix, knn_inx)

        pred_labels = F.softmax(gcn_pred, dim=1)

        edges = []
        scores = []
        local_graph_nodes = local_graph_nodes.long().squeeze().cpu().numpy()
        graph_num = node_feats.size(0)

        for graph_inx in range(graph_num):
            pivot = pivot_inx[graph_inx].int().item()
            nodes = local_graph_nodes[graph_inx]
            for neighbor_inx, neighbor in enumerate(knn_inx[graph_inx]):
                neighbor = neighbor.item()
                edges.append([nodes[pivot], nodes[neighbor]])
                scores.append(pred_labels[graph_inx * (knn_inx.shape[1]) +
                                          neighbor_inx, 1].item())

        edges = np.asarray(edges)
        scores = np.asarray(scores)

        return edges, scores, text_comps

    def get_boundary(self, edges, scores, text_comps):

        boundaries = []
        if edges is not None:
            boundaries = merge_text_comps(edges, scores, text_comps,
                                          self.link_thr)

        results = dict(boundary_result=boundaries)

        return results

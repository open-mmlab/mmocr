# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lanms import merge_quadrangle_n9 as la_nms
from mmcv.ops import RoIAlignRotated
from mmcv.runner import BaseModule
from numpy import ndarray
from torch import Tensor
from torch.nn import init

from mmocr.core import TextDetDataSample
from mmocr.models.textdet.heads import BaseTextDetHead
from mmocr.registry import MODELS
from mmocr.utils.postprocessor_utils import fill_hole


def normalize_adjacent_matrix(mat: ndarray) -> ndarray:
    """Normalize adjacent matrix for GCN. This code was partially adapted from
    https://github.com/GXYM/DRRG licensed under the MIT license.

    Args:
        mat (ndarray): The adjacent matrix.

    returns:
        ndarray: The normalized adjacent matrix.
    """
    assert mat.ndim == 2
    assert mat.shape[0] == mat.shape[1]

    mat = mat + np.eye(mat.shape[0])
    d = np.sum(mat, axis=0)
    d = np.clip(d, 0, None)
    d_inv = np.power(d, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_inv = np.diag(d_inv)
    norm_mat = mat.dot(d_inv).transpose().dot(d_inv)
    return norm_mat


def euclidean_distance_matrix(mat_a: ndarray, mat_b: ndarray) -> ndarray:
    """Calculate the Euclidean distance matrix.

    Args:
        mat_a (ndarray): The point sequence.
        mat_b (ndarray): The point sequence with the same dimensions as mat_a.

    returns:
        ndarray: The Euclidean distance matrix.
    """
    assert mat_a.ndim == 2
    assert mat_b.ndim == 2
    assert mat_a.shape[1] == mat_b.shape[1]

    m = mat_a.shape[0]
    n = mat_b.shape[0]

    mat_a_dots = (mat_a * mat_a).sum(axis=1).reshape(
        (m, 1)) * np.ones(shape=(1, n))
    mat_b_dots = (mat_b * mat_b).sum(axis=1) * np.ones(shape=(m, 1))
    mat_d_squared = mat_a_dots + mat_b_dots - 2 * mat_a.dot(mat_b.T)

    zero_mask = np.less(mat_d_squared, 0.0)
    mat_d_squared[zero_mask] = 0.0
    mat_d = np.sqrt(mat_d_squared)
    return mat_d


def feature_embedding(input_feats: ndarray, out_feat_len: int) -> ndarray:
    """Embed features. This code was partially adapted from
    https://github.com/GXYM/DRRG licensed under the MIT license.

    Args:
        input_feats (ndarray): The input features of shape (N, d), where N is
            the number of nodes in graph, d is the input feature vector length.
        out_feat_len (int): The length of output feature vector.

    Returns:
        ndarray: The embedded features.
    """
    assert input_feats.ndim == 2
    assert isinstance(out_feat_len, int)
    assert out_feat_len >= input_feats.shape[1]

    num_nodes = input_feats.shape[0]
    feat_dim = input_feats.shape[1]
    feat_repeat_times = out_feat_len // feat_dim
    residue_dim = out_feat_len % feat_dim

    if residue_dim > 0:
        embed_wave = np.array([
            np.power(1000, 2.0 * (j // 2) / feat_repeat_times + 1)
            for j in range(feat_repeat_times + 1)
        ]).reshape((feat_repeat_times + 1, 1, 1))
        repeat_feats = np.repeat(
            np.expand_dims(input_feats, axis=0), feat_repeat_times, axis=0)
        residue_feats = np.hstack([
            input_feats[:, 0:residue_dim],
            np.zeros((num_nodes, feat_dim - residue_dim))
        ])
        residue_feats = np.expand_dims(residue_feats, axis=0)
        repeat_feats = np.concatenate([repeat_feats, residue_feats], axis=0)
        embedded_feats = repeat_feats / embed_wave
        embedded_feats[:, 0::2] = np.sin(embedded_feats[:, 0::2])
        embedded_feats[:, 1::2] = np.cos(embedded_feats[:, 1::2])
        embedded_feats = np.transpose(embedded_feats, (1, 0, 2)).reshape(
            (num_nodes, -1))[:, 0:out_feat_len]
    else:
        embed_wave = np.array([
            np.power(1000, 2.0 * (j // 2) / feat_repeat_times)
            for j in range(feat_repeat_times)
        ]).reshape((feat_repeat_times, 1, 1))
        repeat_feats = np.repeat(
            np.expand_dims(input_feats, axis=0), feat_repeat_times, axis=0)
        embedded_feats = repeat_feats / embed_wave
        embedded_feats[:, 0::2] = np.sin(embedded_feats[:, 0::2])
        embedded_feats[:, 1::2] = np.cos(embedded_feats[:, 1::2])
        embedded_feats = np.transpose(embedded_feats, (1, 0, 2)).reshape(
            (num_nodes, -1)).astype(np.float32)

    return embedded_feats


@MODELS.register_module()
class DRRGHead(BaseTextDetHead):
    """The class for DRRG head: `Deep Relational Reasoning Graph Network for
    Arbitrary Shape Text Detection <https://arxiv.org/abs/2003.07493>`_.

    Args:
        in_channels (int): The number of input channels.
        k_at_hops (tuple(int)): The number of i-hop neighbors, i = 1, 2.
            Defaults to (8, 4).
        num_adjacent_linkages (int): The number of linkages when constructing
            adjacent matrix. Defaults to 3.
        node_geo_feat_len (int): The length of embedded geometric feature
            vector of a component. Defaults to 120.
        pooling_scale (float): The spatial scale of rotated RoI-Align. Defaults
            to 1.0.
        pooling_output_size (tuple(int)): The output size of RRoI-Aligning.
            Defaults to (4, 3).
        nms_thr (float): The locality-aware NMS threshold of text components.
            Defaults to 0.3.
        min_width (float): The minimum width of text components. Defaults to
            8.0.
        max_width (float): The maximum width of text components. Defaults to
            24.0.
        comp_shrink_ratio (float): The shrink ratio of text components.
            Defaults to 1.03.
        comp_ratio (float): The reciprocal of aspect ratio of text components.
            Defaults to 0.4.
        comp_score_thr (float): The score threshold of text components.
            Defaults to 0.3.
        text_region_thr (float): The threshold for text region probability map.
            Defaults to 0.2.
        center_region_thr (float): The threshold for text center region
            probability map. Defaults to 0.2.
        center_region_area_thr (int): The threshold for filtering small-sized
            text center region. Defaults to 50.
        local_graph_thr (float): The threshold to filter identical local
            graphs. Defaults to 0.7.
        loss (dict): The config of loss that DRRGHead uses. Defaults to
            ``dict(type='DRRGLoss')``.
        postprocessor (dict): Config of postprocessor for Drrg. Defaults to
            ``dict(type='DrrgPostProcessor', link_thr=0.85)``.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to ``dict(type='Normal',
            override=dict(name='out_conv'), mean=0, std=0.01)``.
    """

    def __init__(
        self,
        in_channels: int,
        k_at_hops: Tuple[int, int] = (8, 4),
        num_adjacent_linkages: int = 3,
        node_geo_feat_len: int = 120,
        pooling_scale: float = 1.0,
        pooling_output_size: Tuple[int, int] = (4, 3),
        nms_thr: float = 0.3,
        min_width: float = 8.0,
        max_width: float = 24.0,
        comp_shrink_ratio: float = 1.03,
        comp_ratio: float = 0.4,
        comp_score_thr: float = 0.3,
        text_region_thr: float = 0.2,
        center_region_thr: float = 0.2,
        center_region_area_thr: int = 50,
        local_graph_thr: float = 0.7,
        loss_module: Dict = dict(type='DRRGLoss'),
        postprocessor: Dict = dict(type='DRRGPostprocessor', link_thr=0.85),
        init_cfg: Optional[Union[Dict, List[Dict]]] = dict(
            type='Normal', override=dict(name='out_conv'), mean=0, std=0.01)
    ) -> None:
        super().__init__(
            loss_module=loss_module,
            postprocessor=postprocessor,
            init_cfg=init_cfg)

        assert isinstance(in_channels, int)
        assert isinstance(k_at_hops, tuple)
        assert isinstance(num_adjacent_linkages, int)
        assert isinstance(node_geo_feat_len, int)
        assert isinstance(pooling_scale, float)
        assert isinstance(pooling_output_size, tuple)
        assert isinstance(comp_shrink_ratio, float)
        assert isinstance(nms_thr, float)
        assert isinstance(min_width, float)
        assert isinstance(max_width, float)
        assert isinstance(comp_ratio, float)
        assert isinstance(comp_score_thr, float)
        assert isinstance(text_region_thr, float)
        assert isinstance(center_region_thr, float)
        assert isinstance(center_region_area_thr, int)
        assert isinstance(local_graph_thr, float)

        self.in_channels = in_channels
        self.out_channels = 6
        self.downsample_ratio = 1.0
        self.k_at_hops = k_at_hops
        self.num_adjacent_linkages = num_adjacent_linkages
        self.node_geo_feat_len = node_geo_feat_len
        self.pooling_scale = pooling_scale
        self.pooling_output_size = pooling_output_size
        self.comp_shrink_ratio = comp_shrink_ratio
        self.nms_thr = nms_thr
        self.min_width = min_width
        self.max_width = max_width
        self.comp_ratio = comp_ratio
        self.comp_score_thr = comp_score_thr
        self.text_region_thr = text_region_thr
        self.center_region_thr = center_region_thr
        self.center_region_area_thr = center_region_area_thr
        self.local_graph_thr = local_graph_thr

        self.out_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0)

        self.graph_train = LocalGraphs(self.k_at_hops,
                                       self.num_adjacent_linkages,
                                       self.node_geo_feat_len,
                                       self.pooling_scale,
                                       self.pooling_output_size,
                                       self.local_graph_thr)

        self.graph_test = ProposalLocalGraphs(
            self.k_at_hops, self.num_adjacent_linkages, self.node_geo_feat_len,
            self.pooling_scale, self.pooling_output_size, self.nms_thr,
            self.min_width, self.max_width, self.comp_shrink_ratio,
            self.comp_ratio, self.comp_score_thr, self.text_region_thr,
            self.center_region_thr, self.center_region_area_thr)

        pool_w, pool_h = self.pooling_output_size
        node_feat_len = (pool_w * pool_h) * (
            self.in_channels + self.out_channels) + self.node_geo_feat_len
        self.gcn = GCN(node_feat_len)

    def loss(
        self, batch_inputs: torch.Tensor,
        batch_data_samples: List[TextDetDataSample]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Loss function.

        Args:
            batch_inputs (Tensor): Shape of :math:`(N, C, H, W)`.
            batch_data_samples (List[TextDetDataSample]): List of data samples.

        Returns:
            tuple(pred_maps, gcn_pred, gt_labels):

            - pred_maps (Tensor): Prediction map with shape
                :math:`(N, 6, H, W)`.
            - gcn_pred (Tensor): Prediction from GCN module, with
                shape :math:`(N, 2)`.
            - gt_labels (Tensor): Ground-truth label of shape
                :math:`(m, n)` where :math:`m * n = N`.
        """
        targets = self.loss_module.get_targets(batch_data_samples)
        gt_comp_attribs = targets[-1]

        pred_maps = self.out_conv(batch_inputs)
        feat_maps = torch.cat([batch_inputs, pred_maps], dim=1)
        node_feats, adjacent_matrices, knn_inds, gt_labels = self.graph_train(
            feat_maps, np.stack(gt_comp_attribs))

        gcn_pred = self.gcn(node_feats, adjacent_matrices, knn_inds)

        return self.loss_module((pred_maps, gcn_pred, gt_labels),
                                batch_data_samples)

    def forward(self, batch_inputs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Run DRRG head in prediction mode, and return the raw tensors only.
        Args:
            batch_inputs (Tensor): Shape of :math:`(1, C, H, W)`.

        Returns:
            tuple: Returns (edge, score, text_comps).

            - edge (ndarray): The edge array of shape :math:`(N_{edges}, 2)`
              where each row is a pair of text component indices
              that makes up an edge in graph.
            - score (ndarray): The score array of shape :math:`(N_{edges},)`,
              corresponding to the edge above.
            - text_comps (ndarray): The text components of shape
              :math:`(M, 9)` where each row corresponds to one box and
              its score: (x1, y1, x2, y2, x3, y3, x4, y4, score).
        """
        pred_maps = self.out_conv(batch_inputs)
        batch_inputs = torch.cat([batch_inputs, pred_maps], dim=1)

        none_flag, graph_data = self.graph_test(pred_maps, batch_inputs)

        (local_graphs_node_feat, adjacent_matrices, pivots_knn_inds,
         pivot_local_graphs, text_comps) = graph_data

        if none_flag:
            return None, None, None

        gcn_pred = self.gcn(local_graphs_node_feat, adjacent_matrices,
                            pivots_knn_inds)
        pred_labels = F.softmax(gcn_pred, dim=1)

        edges = []
        scores = []
        pivot_local_graphs = pivot_local_graphs.long().squeeze().cpu().numpy()

        for pivot_ind, pivot_local_graph in enumerate(pivot_local_graphs):
            pivot = pivot_local_graph[0]
            for k_ind, neighbor_ind in enumerate(pivots_knn_inds[pivot_ind]):
                neighbor = pivot_local_graph[neighbor_ind.item()]
                edges.append([pivot, neighbor])
                scores.append(
                    pred_labels[pivot_ind * pivots_knn_inds.shape[1] + k_ind,
                                1].item())

        edges = np.asarray(edges)
        scores = np.asarray(scores)

        return edges, scores, text_comps


class LocalGraphs:
    """Generate local graphs for GCN to classify the neighbors of a pivot for
    `DRRG: Deep Relational Reasoning Graph Network for Arbitrary Shape Text
    Detection <[https://arxiv.org/abs/2003.07493]>`_.

    This code was partially adapted from
    https://github.com/GXYM/DRRG licensed under the MIT license.

    Args:
        k_at_hops (tuple(int)): The number of i-hop neighbors, i = 1, 2.
        num_adjacent_linkages (int): The number of linkages when constructing
            adjacent matrix.
        node_geo_feat_len (int): The length of embedded geometric feature
            vector of a text component.
        pooling_scale (float): The spatial scale of rotated RoI-Align.
        pooling_output_size (tuple(int)): The output size of rotated RoI-Align.
        local_graph_thr(float): The threshold for filtering out identical local
            graphs.
    """

    def __init__(self, k_at_hops: Tuple[int, int], num_adjacent_linkages: int,
                 node_geo_feat_len: int, pooling_scale: float,
                 pooling_output_size: Sequence[int],
                 local_graph_thr: float) -> None:

        assert len(k_at_hops) == 2
        assert all(isinstance(n, int) for n in k_at_hops)
        assert isinstance(num_adjacent_linkages, int)
        assert isinstance(node_geo_feat_len, int)
        assert isinstance(pooling_scale, float)
        assert all(isinstance(n, int) for n in pooling_output_size)
        assert isinstance(local_graph_thr, float)

        self.k_at_hops = k_at_hops
        self.num_adjacent_linkages = num_adjacent_linkages
        self.node_geo_feat_dim = node_geo_feat_len
        self.pooling = RoIAlignRotated(pooling_output_size, pooling_scale)
        self.local_graph_thr = local_graph_thr

    def generate_local_graphs(self, sorted_dist_inds: ndarray,
                              gt_comp_labels: ndarray
                              ) -> Tuple[List[List[int]], List[List[int]]]:
        """Generate local graphs for GCN to predict which instance a text
        component belongs to.

        Args:
            sorted_dist_inds (ndarray): The complete graph node indices, which
                is sorted according to the Euclidean distance.
            gt_comp_labels(ndarray): The ground truth labels define the
                instance to which the text components (nodes in graphs) belong.

        Returns:
            Tuple(pivot_local_graphs, pivot_knns):

            - pivot_local_graphs (list[list[int]]): The list of local graph
              neighbor indices of pivots.
            - pivot_knns (list[list[int]]): The list of k-nearest neighbor
              indices of pivots.
        """

        assert sorted_dist_inds.ndim == 2
        assert (sorted_dist_inds.shape[0] == sorted_dist_inds.shape[1] ==
                gt_comp_labels.shape[0])

        knn_graph = sorted_dist_inds[:, 1:self.k_at_hops[0] + 1]
        pivot_local_graphs = []
        pivot_knns = []
        for pivot_ind, knn in enumerate(knn_graph):

            local_graph_neighbors = set(knn)

            for neighbor_ind in knn:
                local_graph_neighbors.update(
                    set(sorted_dist_inds[neighbor_ind,
                                         1:self.k_at_hops[1] + 1]))

            local_graph_neighbors.discard(pivot_ind)
            pivot_local_graph = list(local_graph_neighbors)
            pivot_local_graph.insert(0, pivot_ind)
            pivot_knn = [pivot_ind] + list(knn)

            if pivot_ind < 1:
                pivot_local_graphs.append(pivot_local_graph)
                pivot_knns.append(pivot_knn)
            else:
                add_flag = True
                for graph_ind, added_knn in enumerate(pivot_knns):
                    added_pivot_ind = added_knn[0]
                    added_local_graph = pivot_local_graphs[graph_ind]

                    union = len(
                        set(pivot_local_graph[1:]).union(
                            set(added_local_graph[1:])))
                    intersect = len(
                        set(pivot_local_graph[1:]).intersection(
                            set(added_local_graph[1:])))
                    local_graph_iou = intersect / (union + 1e-8)

                    if (local_graph_iou > self.local_graph_thr
                            and pivot_ind in added_knn
                            and gt_comp_labels[added_pivot_ind]
                            == gt_comp_labels[pivot_ind]
                            and gt_comp_labels[pivot_ind] != 0):
                        add_flag = False
                        break
                if add_flag:
                    pivot_local_graphs.append(pivot_local_graph)
                    pivot_knns.append(pivot_knn)

        return pivot_local_graphs, pivot_knns

    def generate_gcn_input(
        self, node_feat_batch: List[Tensor], node_label_batch: List[ndarray],
        local_graph_batch: List[List[List[int]]],
        knn_batch: List[List[List[int]]], sorted_dist_ind_batch: List[ndarray]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Generate graph convolution network input data.

        Args:
            node_feat_batch (List[Tensor]): The batched graph node features.
            node_label_batch (List[ndarray]): The batched text component
                labels.
            local_graph_batch (List[List[List[int]]]): The local graph node
                indices of image batch.
            knn_batch (List[List[List[int]]]): The knn graph node indices of
                image batch.
            sorted_dist_ind_batch (List[ndarray]): The node indices sorted
                according to the Euclidean distance.

        Returns:
            Tuple(local_graphs_node_feat, adjacent_matrices, pivots_knn_inds,
            gt_linkage):

            - local_graphs_node_feat (Tensor): The node features of graph.
            - adjacent_matrices (Tensor): The adjacent matrices of local
              graphs.
            - pivots_knn_inds (Tensor): The k-nearest neighbor indices in
                local graph.
            - gt_linkage (Tensor): The surpervision signal of GCN for linkage
                prediction.
        """
        assert isinstance(node_feat_batch, list)
        assert isinstance(node_label_batch, list)
        assert isinstance(local_graph_batch, list)
        assert isinstance(knn_batch, list)
        assert isinstance(sorted_dist_ind_batch, list)

        num_max_nodes = max(
            len(pivot_local_graph) for pivot_local_graphs in local_graph_batch
            for pivot_local_graph in pivot_local_graphs)

        local_graphs_node_feat = []
        adjacent_matrices = []
        pivots_knn_inds = []
        pivots_gt_linkage = []

        for batch_ind, sorted_dist_inds in enumerate(sorted_dist_ind_batch):
            node_feats = node_feat_batch[batch_ind]
            pivot_local_graphs = local_graph_batch[batch_ind]
            pivot_knns = knn_batch[batch_ind]
            node_labels = node_label_batch[batch_ind]
            device = node_feats.device

            for graph_ind, pivot_knn in enumerate(pivot_knns):
                pivot_local_graph = pivot_local_graphs[graph_ind]
                num_nodes = len(pivot_local_graph)
                pivot_ind = pivot_local_graph[0]
                node2ind_map = {j: i for i, j in enumerate(pivot_local_graph)}

                knn_inds = torch.tensor(
                    [node2ind_map[i] for i in pivot_knn[1:]])
                pivot_feats = node_feats[pivot_ind]
                normalized_feats = node_feats[pivot_local_graph] - pivot_feats

                adjacent_matrix = np.zeros((num_nodes, num_nodes),
                                           dtype=np.float32)
                for node in pivot_local_graph:
                    neighbors = sorted_dist_inds[node,
                                                 1:self.num_adjacent_linkages +
                                                 1]
                    for neighbor in neighbors:
                        if neighbor in pivot_local_graph:

                            adjacent_matrix[node2ind_map[node],
                                            node2ind_map[neighbor]] = 1
                            adjacent_matrix[node2ind_map[neighbor],
                                            node2ind_map[node]] = 1

                adjacent_matrix = normalize_adjacent_matrix(adjacent_matrix)
                pad_adjacent_matrix = torch.zeros(
                    (num_max_nodes, num_max_nodes),
                    dtype=torch.float,
                    device=device)
                pad_adjacent_matrix[:num_nodes, :num_nodes] = torch.from_numpy(
                    adjacent_matrix)

                pad_normalized_feats = torch.cat([
                    normalized_feats,
                    torch.zeros(
                        (num_max_nodes - num_nodes, normalized_feats.shape[1]),
                        dtype=torch.float,
                        device=device)
                ],
                                                 dim=0)

                local_graph_labels = node_labels[pivot_local_graph]
                knn_labels = local_graph_labels[knn_inds]
                link_labels = ((node_labels[pivot_ind] == knn_labels) &
                               (node_labels[pivot_ind] > 0)).astype(np.int64)
                link_labels = torch.from_numpy(link_labels)

                local_graphs_node_feat.append(pad_normalized_feats)
                adjacent_matrices.append(pad_adjacent_matrix)
                pivots_knn_inds.append(knn_inds)
                pivots_gt_linkage.append(link_labels)

        local_graphs_node_feat = torch.stack(local_graphs_node_feat, 0)
        adjacent_matrices = torch.stack(adjacent_matrices, 0)
        pivots_knn_inds = torch.stack(pivots_knn_inds, 0)
        pivots_gt_linkage = torch.stack(pivots_gt_linkage, 0)

        return (local_graphs_node_feat, adjacent_matrices, pivots_knn_inds,
                pivots_gt_linkage)

    def __call__(self, feat_maps: Tensor, comp_attribs: ndarray
                 ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Generate local graphs as GCN input.

        Args:
            feat_maps (Tensor): The feature maps to extract the content
                features of text components.
            comp_attribs (ndarray): The text component attributes.

        Returns:
            Tuple(local_graphs_node_feat, adjacent_matrices, pivots_knn_inds,
            gt_linkage):

            - local_graphs_node_feat (Tensor): The node features of graph.
            - adjacent_matrices (Tensor): The adjacent matrices of local
              graphs.
            - pivots_knn_inds (Tensor): The k-nearest neighbor indices in local
              graph.
            - gt_linkage (Tensor): The surpervision signal of GCN for linkage
              prediction.
        """

        assert isinstance(feat_maps, Tensor)
        assert comp_attribs.ndim == 3
        assert comp_attribs.shape[2] == 8

        sorted_dist_inds_batch = []
        local_graph_batch = []
        knn_batch = []
        node_feat_batch = []
        node_label_batch = []
        device = feat_maps.device

        for batch_ind in range(comp_attribs.shape[0]):
            num_comps = int(comp_attribs[batch_ind, 0, 0])
            comp_geo_attribs = comp_attribs[batch_ind, :num_comps, 1:7]
            node_labels = comp_attribs[batch_ind, :num_comps,
                                       7].astype(np.int32)

            comp_centers = comp_geo_attribs[:, 0:2]
            distance_matrix = euclidean_distance_matrix(
                comp_centers, comp_centers)

            batch_id = np.zeros(
                (comp_geo_attribs.shape[0], 1), dtype=np.float32) * batch_ind
            comp_geo_attribs[:, -2] = np.clip(comp_geo_attribs[:, -2], -1, 1)
            angle = np.arccos(comp_geo_attribs[:, -2]) * np.sign(
                comp_geo_attribs[:, -1])
            angle = angle.reshape((-1, 1))
            rotated_rois = np.hstack(
                [batch_id, comp_geo_attribs[:, :-2], angle])
            rois = torch.from_numpy(rotated_rois).to(device)
            content_feats = self.pooling(feat_maps[batch_ind].unsqueeze(0),
                                         rois)

            content_feats = content_feats.view(content_feats.shape[0],
                                               -1).to(feat_maps.device)
            geo_feats = feature_embedding(comp_geo_attribs,
                                          self.node_geo_feat_dim)
            geo_feats = torch.from_numpy(geo_feats).to(device)
            node_feats = torch.cat([content_feats, geo_feats], dim=-1)

            sorted_dist_inds = np.argsort(distance_matrix, axis=1)
            pivot_local_graphs, pivot_knns = self.generate_local_graphs(
                sorted_dist_inds, node_labels)

            node_feat_batch.append(node_feats)
            node_label_batch.append(node_labels)
            local_graph_batch.append(pivot_local_graphs)
            knn_batch.append(pivot_knns)
            sorted_dist_inds_batch.append(sorted_dist_inds)

        (node_feats, adjacent_matrices, knn_inds, gt_linkage) = \
            self.generate_gcn_input(node_feat_batch,
                                    node_label_batch,
                                    local_graph_batch,
                                    knn_batch,
                                    sorted_dist_inds_batch)

        return node_feats, adjacent_matrices, knn_inds, gt_linkage


class ProposalLocalGraphs:
    """Propose text components and generate local graphs for GCN to classify
    the k-nearest neighbors of a pivot in `DRRG: Deep Relational Reasoning
    Graph Network for Arbitrary Shape Text Detection.

    <https://arxiv.org/abs/2003.07493>`_.

    This code was partially adapted from https://github.com/GXYM/DRRG licensed
    under the MIT license.

    Args:
        k_at_hops (tuple(int)): The number of i-hop neighbors, i = 1, 2.
        num_adjacent_linkages (int): The number of linkages when constructing
            adjacent matrix.
        node_geo_feat_len (int): The length of embedded geometric feature
            vector of a text component.
        pooling_scale (float): The spatial scale of rotated RoI-Align.
        pooling_output_size (tuple(int)): The output size of rotated RoI-Align.
        nms_thr (float): The locality-aware NMS threshold for text components.
        min_width (float): The minimum width of text components.
        max_width (float): The maximum width of text components.
        comp_shrink_ratio (float): The shrink ratio of text components.
        comp_w_h_ratio (float): The width to height ratio of text components.
        comp_score_thr (float): The score threshold of text component.
        text_region_thr (float): The threshold for text region probability map.
        center_region_thr (float): The threshold for text center region
            probability map.
        center_region_area_thr (int): The threshold for filtering small-sized
            text center region.
    """

    def __init__(self, k_at_hops: Tuple[int, int], num_adjacent_linkages: int,
                 node_geo_feat_len: int, pooling_scale: float,
                 pooling_output_size: Sequence[int], nms_thr: float,
                 min_width: float, max_width: float, comp_shrink_ratio: float,
                 comp_w_h_ratio: float, comp_score_thr: float,
                 text_region_thr: float, center_region_thr: float,
                 center_region_area_thr: int) -> None:

        assert len(k_at_hops) == 2
        assert isinstance(k_at_hops, tuple)
        assert isinstance(num_adjacent_linkages, int)
        assert isinstance(node_geo_feat_len, int)
        assert isinstance(pooling_scale, float)
        assert isinstance(pooling_output_size, tuple)
        assert isinstance(nms_thr, float)
        assert isinstance(min_width, float)
        assert isinstance(max_width, float)
        assert isinstance(comp_shrink_ratio, float)
        assert isinstance(comp_w_h_ratio, float)
        assert isinstance(comp_score_thr, float)
        assert isinstance(text_region_thr, float)
        assert isinstance(center_region_thr, float)
        assert isinstance(center_region_area_thr, int)

        self.k_at_hops = k_at_hops
        self.active_connection = num_adjacent_linkages
        self.local_graph_depth = len(self.k_at_hops)
        self.node_geo_feat_dim = node_geo_feat_len
        self.pooling = RoIAlignRotated(pooling_output_size, pooling_scale)
        self.nms_thr = nms_thr
        self.min_width = min_width
        self.max_width = max_width
        self.comp_shrink_ratio = comp_shrink_ratio
        self.comp_w_h_ratio = comp_w_h_ratio
        self.comp_score_thr = comp_score_thr
        self.text_region_thr = text_region_thr
        self.center_region_thr = center_region_thr
        self.center_region_area_thr = center_region_area_thr

    def propose_comps(self, score_map: ndarray, top_height_map: ndarray,
                      bot_height_map: ndarray, sin_map: ndarray,
                      cos_map: ndarray, comp_score_thr: float,
                      min_width: float, max_width: float,
                      comp_shrink_ratio: float,
                      comp_w_h_ratio: float) -> ndarray:
        """Propose text components.

        Args:
            score_map (ndarray): The score map for NMS.
            top_height_map (ndarray): The predicted text height map from each
                pixel in text center region to top sideline.
            bot_height_map (ndarray): The predicted text height map from each
                pixel in text center region to bottom sideline.
            sin_map (ndarray): The predicted sin(theta) map.
            cos_map (ndarray): The predicted cos(theta) map.
            comp_score_thr (float): The score threshold of text component.
            min_width (float): The minimum width of text components.
            max_width (float): The maximum width of text components.
            comp_shrink_ratio (float): The shrink ratio of text components.
            comp_w_h_ratio (float): The width to height ratio of text
                components.

        Returns:
            ndarray: The text components.
        """

        comp_centers = np.argwhere(score_map > comp_score_thr)
        comp_centers = comp_centers[np.argsort(comp_centers[:, 0])]
        y = comp_centers[:, 0]
        x = comp_centers[:, 1]

        top_height = top_height_map[y, x].reshape((-1, 1)) * comp_shrink_ratio
        bot_height = bot_height_map[y, x].reshape((-1, 1)) * comp_shrink_ratio
        sin = sin_map[y, x].reshape((-1, 1))
        cos = cos_map[y, x].reshape((-1, 1))

        top_mid_pts = comp_centers + np.hstack(
            [top_height * sin, top_height * cos])
        bot_mid_pts = comp_centers - np.hstack(
            [bot_height * sin, bot_height * cos])

        width = (top_height + bot_height) * comp_w_h_ratio
        width = np.clip(width, min_width, max_width)
        r = width / 2

        tl = top_mid_pts[:, ::-1] - np.hstack([-r * sin, r * cos])
        tr = top_mid_pts[:, ::-1] + np.hstack([-r * sin, r * cos])
        br = bot_mid_pts[:, ::-1] + np.hstack([-r * sin, r * cos])
        bl = bot_mid_pts[:, ::-1] - np.hstack([-r * sin, r * cos])
        text_comps = np.hstack([tl, tr, br, bl]).astype(np.float32)

        score = score_map[y, x].reshape((-1, 1))
        text_comps = np.hstack([text_comps, score])

        return text_comps

    def propose_comps_and_attribs(self, text_region_map: ndarray,
                                  center_region_map: ndarray,
                                  top_height_map: ndarray,
                                  bot_height_map: ndarray, sin_map: ndarray,
                                  cos_map: ndarray) -> Tuple[ndarray, ndarray]:
        """Generate text components and attributes.

        Args:
            text_region_map (ndarray): The predicted text region probability
                map.
            center_region_map (ndarray): The predicted text center region
                probability map.
            top_height_map (ndarray): The predicted text height map from each
                pixel in text center region to top sideline.
            bot_height_map (ndarray): The predicted text height map from each
                pixel in text center region to bottom sideline.
            sin_map (ndarray): The predicted sin(theta) map.
            cos_map (ndarray): The predicted cos(theta) map.

        Returns:
            tuple(ndarray, ndarray):

            - comp_attribs (ndarray): The text component attributes.
            - text_comps (ndarray): The text components.
        """

        assert (text_region_map.shape == center_region_map.shape ==
                top_height_map.shape == bot_height_map.shape == sin_map.shape
                == cos_map.shape)
        text_mask = text_region_map > self.text_region_thr
        center_region_mask = (center_region_map >
                              self.center_region_thr) * text_mask

        scale = np.sqrt(1.0 / (sin_map**2 + cos_map**2 + 1e-8))
        sin_map, cos_map = sin_map * scale, cos_map * scale

        center_region_mask = fill_hole(center_region_mask)
        center_region_contours, _ = cv2.findContours(
            center_region_mask.astype(np.uint8), cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE)

        mask_sz = center_region_map.shape
        comp_list = []
        for contour in center_region_contours:
            current_center_mask = np.zeros(mask_sz)
            cv2.drawContours(current_center_mask, [contour], -1, 1, -1)
            if current_center_mask.sum() <= self.center_region_area_thr:
                continue
            score_map = text_region_map * current_center_mask

            text_comps = self.propose_comps(score_map, top_height_map,
                                            bot_height_map, sin_map, cos_map,
                                            self.comp_score_thr,
                                            self.min_width, self.max_width,
                                            self.comp_shrink_ratio,
                                            self.comp_w_h_ratio)

            text_comps = la_nms(text_comps, self.nms_thr)
            text_comp_mask = np.zeros(mask_sz)
            text_comp_boxes = text_comps[:, :8].reshape(
                (-1, 4, 2)).astype(np.int32)

            cv2.drawContours(text_comp_mask, text_comp_boxes, -1, 1, -1)
            if (text_comp_mask * text_mask).sum() < text_comp_mask.sum() * 0.5:
                continue
            if text_comps.shape[-1] > 0:
                comp_list.append(text_comps)

        if len(comp_list) <= 0:
            return None, None

        text_comps = np.vstack(comp_list)
        text_comp_boxes = text_comps[:, :8].reshape((-1, 4, 2))
        centers = np.mean(text_comp_boxes, axis=1).astype(np.int32)
        x = centers[:, 0]
        y = centers[:, 1]

        scores = []
        for text_comp_box in text_comp_boxes:
            text_comp_box[:, 0] = np.clip(text_comp_box[:, 0], 0,
                                          mask_sz[1] - 1)
            text_comp_box[:, 1] = np.clip(text_comp_box[:, 1], 0,
                                          mask_sz[0] - 1)
            min_coord = np.min(text_comp_box, axis=0).astype(np.int32)
            max_coord = np.max(text_comp_box, axis=0).astype(np.int32)
            text_comp_box = text_comp_box - min_coord
            box_sz = (max_coord - min_coord + 1)
            temp_comp_mask = np.zeros((box_sz[1], box_sz[0]), dtype=np.uint8)
            cv2.fillPoly(temp_comp_mask, [text_comp_box.astype(np.int32)], 1)
            temp_region_patch = text_region_map[min_coord[1]:(max_coord[1] +
                                                              1),
                                                min_coord[0]:(max_coord[0] +
                                                              1)]
            score = cv2.mean(temp_region_patch, temp_comp_mask)[0]
            scores.append(score)
        scores = np.array(scores).reshape((-1, 1))
        text_comps = np.hstack([text_comps[:, :-1], scores])

        h = top_height_map[y, x].reshape(
            (-1, 1)) + bot_height_map[y, x].reshape((-1, 1))
        w = np.clip(h * self.comp_w_h_ratio, self.min_width, self.max_width)
        sin = sin_map[y, x].reshape((-1, 1))
        cos = cos_map[y, x].reshape((-1, 1))

        x = x.reshape((-1, 1))
        y = y.reshape((-1, 1))
        comp_attribs = np.hstack([x, y, h, w, cos, sin])

        return comp_attribs, text_comps

    def generate_local_graphs(self, sorted_dist_inds: ndarray,
                              node_feats: Tensor
                              ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Generate local graphs and graph convolution network input data.

        Args:
            sorted_dist_inds (ndarray): The node indices sorted according to
                the Euclidean distance.
            node_feats (tensor): The features of nodes in graph.

        Returns:
            Tuple(local_graphs_node_feat, adjacent_matrices, pivots_knn_inds,
            pivots_local_graphs):

            - local_graphs_node_feats (tensor): The features of nodes in local
              graphs.
            - adjacent_matrices (tensor): The adjacent matrices.
            - pivots_knn_inds (tensor): The k-nearest neighbor indices in
              local graphs.
            - pivots_local_graphs (tensor): The indices of nodes in local
              graphs.
        """

        assert sorted_dist_inds.ndim == 2
        assert (sorted_dist_inds.shape[0] == sorted_dist_inds.shape[1] ==
                node_feats.shape[0])

        knn_graph = sorted_dist_inds[:, 1:self.k_at_hops[0] + 1]
        pivot_local_graphs = []
        pivot_knns = []
        device = node_feats.device

        for pivot_ind, knn in enumerate(knn_graph):

            local_graph_neighbors = set(knn)

            for neighbor_ind in knn:
                local_graph_neighbors.update(
                    set(sorted_dist_inds[neighbor_ind,
                                         1:self.k_at_hops[1] + 1]))

            local_graph_neighbors.discard(pivot_ind)
            pivot_local_graph = list(local_graph_neighbors)
            pivot_local_graph.insert(0, pivot_ind)
            pivot_knn = [pivot_ind] + list(knn)

            pivot_local_graphs.append(pivot_local_graph)
            pivot_knns.append(pivot_knn)

        num_max_nodes = max(
            len(pivot_local_graph) for pivot_local_graph in pivot_local_graphs)

        local_graphs_node_feat = []
        adjacent_matrices = []
        pivots_knn_inds = []
        pivots_local_graphs = []

        for graph_ind, pivot_knn in enumerate(pivot_knns):
            pivot_local_graph = pivot_local_graphs[graph_ind]
            num_nodes = len(pivot_local_graph)
            pivot_ind = pivot_local_graph[0]
            node2ind_map = {j: i for i, j in enumerate(pivot_local_graph)}

            knn_inds = torch.tensor([node2ind_map[i]
                                     for i in pivot_knn[1:]]).long().to(device)
            pivot_feats = node_feats[pivot_ind]
            normalized_feats = node_feats[pivot_local_graph] - pivot_feats

            adjacent_matrix = np.zeros((num_nodes, num_nodes))
            for node in pivot_local_graph:
                neighbors = sorted_dist_inds[node,
                                             1:self.active_connection + 1]
                for neighbor in neighbors:
                    if neighbor in pivot_local_graph:
                        adjacent_matrix[node2ind_map[node],
                                        node2ind_map[neighbor]] = 1
                        adjacent_matrix[node2ind_map[neighbor],
                                        node2ind_map[node]] = 1

            adjacent_matrix = normalize_adjacent_matrix(adjacent_matrix)
            pad_adjacent_matrix = torch.zeros((num_max_nodes, num_max_nodes),
                                              dtype=torch.float,
                                              device=device)
            pad_adjacent_matrix[:num_nodes, :num_nodes] = torch.from_numpy(
                adjacent_matrix)

            pad_normalized_feats = torch.cat([
                normalized_feats,
                torch.zeros(
                    (num_max_nodes - num_nodes, normalized_feats.shape[1]),
                    dtype=torch.float,
                    device=device)
            ],
                                             dim=0)

            local_graph_nodes = torch.tensor(pivot_local_graph)
            local_graph_nodes = torch.cat([
                local_graph_nodes,
                torch.zeros(num_max_nodes - num_nodes, dtype=torch.long)
            ],
                                          dim=-1)

            local_graphs_node_feat.append(pad_normalized_feats)
            adjacent_matrices.append(pad_adjacent_matrix)
            pivots_knn_inds.append(knn_inds)
            pivots_local_graphs.append(local_graph_nodes)

        local_graphs_node_feat = torch.stack(local_graphs_node_feat, 0)
        adjacent_matrices = torch.stack(adjacent_matrices, 0)
        pivots_knn_inds = torch.stack(pivots_knn_inds, 0)
        pivots_local_graphs = torch.stack(pivots_local_graphs, 0)

        return (local_graphs_node_feat, adjacent_matrices, pivots_knn_inds,
                pivots_local_graphs)

    def __call__(self, preds: Tensor, feat_maps: Tensor
                 ) -> Tuple[bool, Tensor, Tensor, Tensor, Tensor, ndarray]:
        """Generate local graphs and graph convolutional network input data.

        Args:
            preds (tensor): The predicted maps.
            feat_maps (tensor): The feature maps to extract content feature of
                text components.

        Returns:
            Tuple(none_flag, local_graphs_node_feat, adjacent_matrices,
            pivots_knn_inds, pivots_local_graphs, text_comps):

            - none_flag (bool): The flag showing whether the number of proposed
              text components is 0.
            - local_graphs_node_feats (tensor): The features of nodes in local
              graphs.
            - adjacent_matrices (tensor): The adjacent matrices.
            - pivots_knn_inds (tensor): The k-nearest neighbor indices in
              local graphs.
            - pivots_local_graphs (tensor): The indices of nodes in local
              graphs.
            - text_comps (ndarray): The predicted text components.
        """

        if preds.ndim == 4:
            assert preds.shape[0] == 1
            preds = torch.squeeze(preds)
        pred_text_region = torch.sigmoid(preds[0]).data.cpu().numpy()
        pred_center_region = torch.sigmoid(preds[1]).data.cpu().numpy()
        pred_sin_map = preds[2].data.cpu().numpy()
        pred_cos_map = preds[3].data.cpu().numpy()
        pred_top_height_map = preds[4].data.cpu().numpy()
        pred_bot_height_map = preds[5].data.cpu().numpy()
        device = preds.device

        comp_attribs, text_comps = self.propose_comps_and_attribs(
            pred_text_region, pred_center_region, pred_top_height_map,
            pred_bot_height_map, pred_sin_map, pred_cos_map)

        if comp_attribs is None or len(comp_attribs) < 2:
            none_flag = True
            return none_flag, (0, 0, 0, 0, 0)

        comp_centers = comp_attribs[:, 0:2]
        distance_matrix = euclidean_distance_matrix(comp_centers, comp_centers)

        geo_feats = feature_embedding(comp_attribs, self.node_geo_feat_dim)
        geo_feats = torch.from_numpy(geo_feats).to(preds.device)

        batch_id = np.zeros((comp_attribs.shape[0], 1), dtype=np.float32)
        comp_attribs = comp_attribs.astype(np.float32)
        angle = np.arccos(comp_attribs[:, -2]) * np.sign(comp_attribs[:, -1])
        angle = angle.reshape((-1, 1))
        rotated_rois = np.hstack([batch_id, comp_attribs[:, :-2], angle])
        rois = torch.from_numpy(rotated_rois).to(device)

        content_feats = self.pooling(feat_maps, rois)
        content_feats = content_feats.view(content_feats.shape[0],
                                           -1).to(device)
        node_feats = torch.cat([content_feats, geo_feats], dim=-1)

        sorted_dist_inds = np.argsort(distance_matrix, axis=1)
        (local_graphs_node_feat, adjacent_matrices, pivots_knn_inds,
         pivots_local_graphs) = self.generate_local_graphs(
             sorted_dist_inds, node_feats)

        none_flag = False
        return none_flag, (local_graphs_node_feat, adjacent_matrices,
                           pivots_knn_inds, pivots_local_graphs, text_comps)


class GraphConv(BaseModule):
    """Graph convolutional neural network.

    Args:
        in_dim (int): The number of input channels.
        out_dim (int): The number of output channels.
    """

    class MeanAggregator(BaseModule):
        """Mean aggregator for graph convolutional network."""

        def forward(self, features: Tensor, A: Tensor) -> Tensor:
            """Forward function."""
            x = torch.bmm(A, features)
            return x

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.FloatTensor(in_dim * 2, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        init.xavier_uniform_(self.weight)
        init.constant_(self.bias, 0)
        self.aggregator = self.MeanAggregator()

    def forward(self, features: Tensor, A: Tensor) -> Tensor:
        """Forward function."""
        _, _, d = features.shape
        assert d == self.in_dim
        agg_feats = self.aggregator(features, A)
        cat_feats = torch.cat([features, agg_feats], dim=2)
        out = torch.einsum('bnd,df->bnf', cat_feats, self.weight)
        out = F.relu(out + self.bias)
        return out


class GCN(BaseModule):
    """Graph convolutional network for clustering. This was from repo
    https://github.com/Zhongdao/gcn_clustering licensed under the MIT license.

    Args:
        feat_len (int): The input node feature length.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 feat_len: int,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.bn0 = nn.BatchNorm1d(feat_len, affine=False).float()
        self.conv1 = GraphConv(feat_len, 512)
        self.conv2 = GraphConv(512, 256)
        self.conv3 = GraphConv(256, 128)
        self.conv4 = GraphConv(128, 64)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.PReLU(32), nn.Linear(32, 2))

    def forward(self, node_feats: Tensor, adj_mats: Tensor,
                knn_inds: Tensor) -> Tensor:
        """Forward function.

        Args:
            local_graphs_node_feat (Tensor): The node features of graph.
            adjacent_matrices (Tensor): The adjacent matrices of local
                graphs.
            pivots_knn_inds (Tensor): The k-nearest neighbor indices in
                local graph.

        Returns:
            Tensor: The output feature.
        """

        num_local_graphs, num_max_nodes, feat_len = node_feats.shape

        node_feats = node_feats.view(-1, feat_len)
        node_feats = self.bn0(node_feats)
        node_feats = node_feats.view(num_local_graphs, num_max_nodes, feat_len)

        node_feats = self.conv1(node_feats, adj_mats)
        node_feats = self.conv2(node_feats, adj_mats)
        node_feats = self.conv3(node_feats, adj_mats)
        node_feats = self.conv4(node_feats, adj_mats)
        k = knn_inds.size(-1)
        mid_feat_len = node_feats.size(-1)
        edge_feat = torch.zeros((num_local_graphs, k, mid_feat_len),
                                device=node_feats.device)
        for graph_ind in range(num_local_graphs):
            edge_feat[graph_ind, :, :] = node_feats[graph_ind,
                                                    knn_inds[graph_ind]]
        edge_feat = edge_feat.view(-1, mid_feat_len)
        pred = self.classifier(edge_feat)

        return pred

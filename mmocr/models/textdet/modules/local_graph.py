import numpy as np
import torch
from mmcv.ops import RoIAlignRotated

from .utils import (euclidean_distance_matrix, feature_embedding,
                    normalize_adjacent_matrix)


class LocalGraphs(object):
    """Generate local graphs for GCN to classify the neighbors of a pivot for
    DRRG: Deep Relational Reasoning Graph Network for Arbitrary Shape Text
    Detection.

    [https://arxiv.org/abs/2003.07493]. This code was partially adapted from
    https://github.com/GXYM/DRRG.

    Args:
        k_at_hops (tuple(int)): The number of h-hop neighbors.
        active_connection (int): The number of neighbors deem as linked to a
            pivot.
        node_geo_feat_dim (int): The dimension of embedded geometric features
            of a component.
        pooling_scale (float): The spatial scale of RRoI-Aligning.
        pooling_output_size (tuple(int)): The size of RRoI-Aligning output.
        local_graph_thr(float): The threshold for filtering out identical local
            graphs.
    """

    def __init__(self, k_at_hops, active_connection, node_geo_feat_dim,
                 pooling_scale, pooling_output_size, local_graph_thr):

        assert len(k_at_hops) == 2
        assert all(isinstance(n, int) for n in k_at_hops)
        assert isinstance(active_connection, int)
        assert isinstance(node_geo_feat_dim, int)
        assert isinstance(pooling_scale, float)
        assert all(isinstance(n, int) for n in pooling_output_size)
        assert isinstance(local_graph_thr, float)

        self.k_at_hops = k_at_hops
        self.active_connection = active_connection
        self.node_geo_feat_dim = node_geo_feat_dim
        self.pooling = RoIAlignRotated(pooling_output_size, pooling_scale)
        self.local_graph_thr = local_graph_thr

    def generate_local_graphs(self, sorted_dist_inds, gt_comp_labels):
        """Generate local graphs for GCN to predict which instance a text
        component belongs to.

        Args:
            sorted_dist_inds (ndarray): The complete graph node indexes, which
                is sorted according to euclidean distance.
            gt_comp_labels(ndarray): The ground truth labels define which
                instance text components (nodes in graphs) belong to.

        Returns:
            pivot_local_graphs(list[list[int]]): The list of local graph
                neighbor indexes of pivots.
            pivot_knns(list[list[int]]): The list of k nearest neighbor indexes
                of pivots.
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

    def generate_gcn_input(self, node_feat_batch, node_label_batch,
                           local_graph_batch, knn_batch,
                           sorted_dist_ind_batch):
        """Generate graph convolution network input data.

        Args:
            node_feat_batch (List[Tensor]): The batched graph node features.
            node_label_batch (List[ndarray]): The batched text component
                labels.
            local_graph_batch (List[List[list[int]]]): The local graph node
                indices of image batch.
            knn_batch (List[List[list[int]]]): The knn graph node indices of
                image batch.
            sorted_dist_ind_batch (list[ndarray]): The node indices sorted
                according to euclidean distance.

        Returns:
            local_graphs_node_feat (Tensor): The node features of graph.
            adjacent_matrices (Tensor): The adjacent matrices of local graphs.
            pivots_knn_inds (Tensor): The k-nearest neighbor indices in
                local graph.
            gt_linkage (Tensor): The surpervision signal of GCN for linkage
                prediction.
        """
        assert isinstance(node_feat_batch, list)
        assert isinstance(node_label_batch, list)
        assert isinstance(local_graph_batch, list)
        assert isinstance(knn_batch, list)
        assert isinstance(sorted_dist_ind_batch, list)

        max_node_num = max([
            len(pivot_local_graph) for pivot_local_graphs in local_graph_batch
            for pivot_local_graph in pivot_local_graphs
        ])

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
                node_num = len(pivot_local_graph)
                pivot_ind = pivot_local_graph[0]
                node2ind_map = {j: i for i, j in enumerate(pivot_local_graph)}

                knn_inds = torch.tensor(
                    [node2ind_map[i] for i in pivot_knn[1:]])
                pivot_feats = node_feats[pivot_ind]
                normalized_feats = node_feats[pivot_local_graph] - pivot_feats

                adjacent_matrix = np.zeros((node_num, node_num),
                                           dtype=np.float32)
                for node in pivot_local_graph:
                    neighbors = sorted_dist_inds[node,
                                                 1:self.active_connection + 1]
                    for neighbor in neighbors:
                        if neighbor in pivot_local_graph:

                            adjacent_matrix[node2ind_map[node],
                                            node2ind_map[neighbor]] = 1
                            adjacent_matrix[node2ind_map[neighbor],
                                            node2ind_map[node]] = 1

                adjacent_matrix = normalize_adjacent_matrix(
                    adjacent_matrix, mode='DAD')
                pad_adjacent_matrix = torch.zeros((max_node_num, max_node_num),
                                                  dtype=torch.float,
                                                  device=device)
                pad_adjacent_matrix[:node_num, :node_num] = adjacent_matrix

                pad_normalized_feats = torch.cat([
                    normalized_feats,
                    torch.zeros(
                        (max_node_num - node_num, normalized_feats.shape[1]),
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

    def __call__(self, feat_maps, comp_attribs):
        """Generate local graphs as GCN input.

        Args:
            feat_maps (Tensor): The feature maps to extract content feature of
                text components.
            comp_attribs (ndarray): The text components attributes.

        Returns:
            local_graphs_node_feat (Tensor): The node features of graph.
            adjacent_matrices (Tensor): The adjacent matrices of local graphs.
            pivots_knn_inds (Tensor): The k-nearest neighbor indices in local
                graph.
            gt_linkage (Tensor): The surpervision signal of GCN for linkage
                prediction.
        """

        assert isinstance(feat_maps, torch.Tensor)
        assert comp_attribs.ndim == 3
        assert comp_attribs.shape[2] == 8

        sorted_dist_inds_batch = []
        local_graph_batch = []
        knn_batch = []
        node_feat_batch = []
        node_label_batch = []
        device = feat_maps.device

        for batch_ind in range(comp_attribs.shape[0]):
            comp_num = int(comp_attribs[batch_ind, 0, 0])
            comp_geo_attribs = comp_attribs[batch_ind, :comp_num, 1:7]
            node_labels = comp_attribs[batch_ind, :comp_num,
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

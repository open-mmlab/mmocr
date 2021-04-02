import numpy as np
import torch

from mmocr.models.utils import RROIAlign
from .utils import (embed_geo_feats, euclidean_distance_matrix,
                    normalize_adjacent_matrix)


class LocalGraphs:
    """Generate local graphs for GCN to predict which instance a text component
    belongs to. This was partially adapted from https://github.com/GXYM/DRRG:
    Deep Relational Reasoning Graph Network for Arbitrary Shape Text Detection.

    [https://arxiv.org/abs/2003.07493]

    Args:
        k_at_hops (tuple(int)): The number of h-hop neighbors.
        active_connection (int): The number of neighbors deem as linked to a
            pivot.
        node_geo_feat_dim (int): The dimension of embedded geometric features
            of a component.
        pooling_scale (float): The spatial scale of RRoI-Aligning.
        pooling_output_size (tuple(int)): The size of RRoI-Aligning output.
        local_graph_filter_thr (float): The threshold to filter out identical
            local graphs.
    """

    def __init__(self, k_at_hops, active_connection, node_geo_feat_dim,
                 pooling_scale, pooling_output_size, local_graph_filter_thr):

        assert isinstance(k_at_hops, tuple)
        assert isinstance(active_connection, int)
        assert isinstance(node_geo_feat_dim, int)
        assert isinstance(pooling_scale, float)
        assert isinstance(pooling_output_size, tuple)
        assert isinstance(local_graph_filter_thr, float)

        self.k_at_hops = k_at_hops
        self.local_graph_depth = len(self.k_at_hops)
        self.active_connection = active_connection
        self.node_geo_feat_dim = node_geo_feat_dim
        self.pooling = RROIAlign(pooling_output_size, pooling_scale)
        self.local_graph_filter_thr = local_graph_filter_thr

    def generate_local_graphs(self, sorted_complete_graph, gt_belong_labels):
        """Generate local graphs for GCN to predict which instance a text
        component belongs to.

        Args:
            sorted_complete_graph (ndarray): The complete graph where nodes are
                sorted according to their Euclidean distance.
            gt_belong_labels (ndarray): The ground truth labels define which
                instance text components (nodes in graphs) belong to.

        Returns:
            local_graph_node_list (list): The list of local graph neighbors of
                pivots.
            knn_graph_neighbor_list (list): The list of k nearest neighbors of
                pivots.
        """

        assert sorted_complete_graph.ndim == 2
        assert (sorted_complete_graph.shape[0] ==
                sorted_complete_graph.shape[1] == gt_belong_labels.shape[0])

        knn_graphs = sorted_complete_graph[:, :self.k_at_hops[0] + 1]
        local_graph_node_list = list()
        knn_graph_neighbor_list = list()
        for pivot_inx, knn_graph in enumerate(knn_graphs):

            h_hop_neighbor_list = list()
            one_hop_neighbors = set(knn_graph[1:])
            h_hop_neighbor_list.append(one_hop_neighbors)

            for hop_inx in range(1, self.local_graph_depth):
                h_hop_neighbor_list.append(set())
                for last_hop_neighbor_inx in h_hop_neighbor_list[-2]:
                    h_hop_neighbor_list[-1].update(
                        set(sorted_complete_graph[last_hop_neighbor_inx]
                            [1:self.k_at_hops[hop_inx] + 1]))

            hops_neighbor_set = set(
                [node for hop in h_hop_neighbor_list for node in hop])
            hops_neighbor_list = list(hops_neighbor_set)
            hops_neighbor_list.insert(0, pivot_inx)

            if pivot_inx < 1:
                local_graph_node_list.append(hops_neighbor_list)
                knn_graph_neighbor_list.append(one_hop_neighbors)
            else:
                add_flag = True
                for graph_inx in range(len(knn_graph_neighbor_list)):
                    knn_graph_neighbors = knn_graph_neighbor_list[graph_inx]
                    local_graph_nodes = local_graph_node_list[graph_inx]

                    node_union_num = len(
                        list(
                            set(knn_graph_neighbors).union(
                                set(one_hop_neighbors))))
                    node_intersect_num = len(
                        list(
                            set(knn_graph_neighbors).intersection(
                                set(one_hop_neighbors))))
                    one_hop_iou = node_intersect_num / (node_union_num + 1e-8)

                    if (one_hop_iou > self.local_graph_filter_thr
                            and pivot_inx in knn_graph_neighbors
                            and gt_belong_labels[local_graph_nodes[0]]
                            == gt_belong_labels[pivot_inx]
                            and gt_belong_labels[local_graph_nodes[0]] != 0):
                        add_flag = False
                        break
                if add_flag:
                    local_graph_node_list.append(hops_neighbor_list)
                    knn_graph_neighbor_list.append(one_hop_neighbors)

        return local_graph_node_list, knn_graph_neighbor_list

    def generate_gcn_input(self, node_feat_batch, belong_label_batch,
                           local_graph_node_batch, knn_graph_neighbor_batch,
                           sorted_complete_graph):
        """Generate graph convolution network input data.

        Args:
            node_feat_batch (List[Tensor]): The node feature batch.
            belong_label_batch (List[ndarray]): The text component belong label
                batch.
            local_graph_node_batch (List[List[list]]): The local graph
                neighbors batch.
            knn_graph_neighbor_batch (List[List[set]]): The knn graph neighbor
                batch.
            sorted_complete_graph (List[ndarray]): The complete graph sorted
                according to the Euclidean distance.

        Returns:
            node_feat_batch_tensor (Tensor): The node features of Graph
                Convolutional Network (GCN).
            adjacent_mat_batch_tensor (Tensor): The adjacent matrices.
            knn_inx_batch_tensor (Tensor): The indices of k nearest neighbors.
            gt_linkage_batch_tensor (Tensor): The surpervision signal of GCN
                for linkage prediction.
        """

        assert isinstance(node_feat_batch, list)
        assert isinstance(belong_label_batch, list)
        assert isinstance(local_graph_node_batch, list)
        assert isinstance(knn_graph_neighbor_batch, list)
        assert isinstance(sorted_complete_graph, list)

        max_graph_node_num = max([
            len(local_graph_nodes)
            for local_graph_node_list in local_graph_node_batch
            for local_graph_nodes in local_graph_node_list
        ])

        node_feat_batch_list = list()
        adjacent_matrix_batch_list = list()
        knn_inx_batch_list = list()
        gt_linkage_batch_list = list()

        for batch_inx, sorted_neighbors in enumerate(sorted_complete_graph):
            node_feats = node_feat_batch[batch_inx]
            local_graph_list = local_graph_node_batch[batch_inx]
            knn_graph_neighbor_list = knn_graph_neighbor_batch[batch_inx]
            belong_labels = belong_label_batch[batch_inx]

            for graph_inx in range(len(local_graph_list)):
                local_graph_nodes = local_graph_list[graph_inx]
                local_graph_node_num = len(local_graph_nodes)
                pivot_inx = local_graph_nodes[0]
                knn_graph_neighbors = knn_graph_neighbor_list[graph_inx]
                node_to_graph_inx = {
                    j: i
                    for i, j in enumerate(local_graph_nodes)
                }

                knn_inx_in_local_graph = torch.tensor(
                    [node_to_graph_inx[i] for i in knn_graph_neighbors],
                    dtype=torch.long)
                pivot_feats = node_feats[torch.tensor(
                    pivot_inx, dtype=torch.long)]
                normalized_feats = node_feats[torch.tensor(
                    local_graph_nodes, dtype=torch.long)] - pivot_feats

                adjacent_matrix = np.zeros(
                    (local_graph_node_num, local_graph_node_num))
                pad_normalized_feats = torch.cat([
                    normalized_feats,
                    torch.zeros(max_graph_node_num - local_graph_node_num,
                                normalized_feats.shape[1]).to(
                                    node_feats.device)
                ],
                                                 dim=0)

                for node in local_graph_nodes:
                    neighbors = sorted_neighbors[node,
                                                 1:self.active_connection + 1]
                    for neighbor in neighbors:
                        if neighbor in local_graph_nodes:
                            adjacent_matrix[node_to_graph_inx[node],
                                            node_to_graph_inx[neighbor]] = 1
                            adjacent_matrix[node_to_graph_inx[neighbor],
                                            node_to_graph_inx[node]] = 1

                adjacent_matrix = normalize_adjacent_matrix(
                    adjacent_matrix, type='DAD')
                adjacent_matrix_tensor = torch.zeros(max_graph_node_num,
                                                     max_graph_node_num).to(
                                                         node_feats.device)
                adjacent_matrix_tensor[:local_graph_node_num, :
                                       local_graph_node_num] = adjacent_matrix

                local_graph_labels = torch.from_numpy(
                    belong_labels[local_graph_nodes]).type(torch.long)
                knn_labels = local_graph_labels[knn_inx_in_local_graph]
                edge_labels = ((belong_labels[pivot_inx] == knn_labels)
                               & (belong_labels[pivot_inx] > 0)).long()

                node_feat_batch_list.append(pad_normalized_feats)
                adjacent_matrix_batch_list.append(adjacent_matrix_tensor)
                knn_inx_batch_list.append(knn_inx_in_local_graph)
                gt_linkage_batch_list.append(edge_labels)

        node_feat_batch_tensor = torch.stack(node_feat_batch_list, 0)
        adjacent_mat_batch_tensor = torch.stack(adjacent_matrix_batch_list, 0)
        knn_inx_batch_tensor = torch.stack(knn_inx_batch_list, 0)
        gt_linkage_batch_tensor = torch.stack(gt_linkage_batch_list, 0)

        return (node_feat_batch_tensor, adjacent_mat_batch_tensor,
                knn_inx_batch_tensor, gt_linkage_batch_tensor)

    def __call__(self, feat_maps, comp_attribs):
        """Generate local graphs.

        Args:
            feat_maps (Tensor): The feature maps to propose node features in
                graph.
            comp_attribs (ndarray): The text components attributes.

        Returns:
            node_feats_batch (Tensor): The node features of Graph Convolutional
                Network(GCN).
            adjacent_matrices_batch (Tensor): The adjacent matrices.
            knn_inx_batch (Tensor): The indices of k nearest neighbors.
            gt_linkage_batch (Tensor): The surpervision signal of GCN for
                linkage prediction.
        """

        assert isinstance(feat_maps, torch.Tensor)
        assert comp_attribs.shape[2] == 8

        dist_sort_graph_batch_list = []
        local_graph_node_batch_list = []
        knn_graph_neighbor_batch_list = []
        node_feature_batch_list = []
        belong_label_batch_list = []

        for batch_inx in range(comp_attribs.shape[0]):
            comp_num = int(comp_attribs[batch_inx, 0, 0])
            comp_geo_attribs = comp_attribs[batch_inx, :comp_num, 1:7]
            node_belong_labels = comp_attribs[batch_inx, :comp_num,
                                              7].astype(np.int32)

            comp_centers = comp_geo_attribs[:, 0:2]
            distance_matrix = euclidean_distance_matrix(
                comp_centers, comp_centers)

            graph_node_geo_feats = embed_geo_feats(comp_geo_attribs,
                                                   self.node_geo_feat_dim)
            graph_node_geo_feats = torch.from_numpy(
                graph_node_geo_feats).float().to(feat_maps.device)

            batch_id = np.zeros(
                (comp_geo_attribs.shape[0], 1), dtype=np.float32) * batch_inx
            text_comps = np.hstack(
                (batch_id, comp_geo_attribs.astype(np.float32)))
            text_comps = torch.from_numpy(text_comps).float().to(
                feat_maps.device)

            comp_content_feats = self.pooling(
                feat_maps[batch_inx].unsqueeze(0), text_comps)
            comp_content_feats = comp_content_feats.view(
                comp_content_feats.shape[0], -1).to(feat_maps.device)
            node_feats = torch.cat((comp_content_feats, graph_node_geo_feats),
                                   dim=-1)

            dist_sort_complete_graph = np.argsort(distance_matrix, axis=1)
            (local_graph_nodes,
             knn_graph_neighbors) = self.generate_local_graphs(
                 dist_sort_complete_graph, node_belong_labels)

            node_feature_batch_list.append(node_feats)
            belong_label_batch_list.append(node_belong_labels)
            local_graph_node_batch_list.append(local_graph_nodes)
            knn_graph_neighbor_batch_list.append(knn_graph_neighbors)
            dist_sort_graph_batch_list.append(dist_sort_complete_graph)

        (node_feats_batch, adjacent_matrices_batch, knn_inx_batch,
         gt_linkage_batch) = \
            self.generate_gcn_input(node_feature_batch_list,
                                    belong_label_batch_list,
                                    local_graph_node_batch_list,
                                    knn_graph_neighbor_batch_list,
                                    dist_sort_graph_batch_list)

        return (node_feats_batch, adjacent_matrices_batch, knn_inx_batch,
                gt_linkage_batch)

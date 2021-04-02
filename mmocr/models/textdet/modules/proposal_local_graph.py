import cv2
import numpy as np
import torch

# from mmocr.models.textdet.postprocess import la_nms
from mmocr.models.utils import RROIAlign
from .utils import (embed_geo_feats, euclidean_distance_matrix,
                    normalize_adjacent_matrix)


class ProposalLocalGraphs:
    """Propose text components and generate local graphs. This was partially
    adapted from https://github.com/GXYM/DRRG: Deep Relational Reasoning Graph
    Network for Arbitrary Shape Text Detection.

    [https://arxiv.org/abs/2003.07493]

    Args:
        k_at_hops (tuple(int)): The number of i-hop neighbors,
            i = 1, 2, ..., h.
        active_connection (int): The number of two hop neighbors deem as linked
            to a pivot.
        node_geo_feat_dim (int): The dimension of embedded geometric features
            of a component.
        pooling_scale (float): The spatial scale of RRoI-Aligning.
        pooling_output_size (tuple(int)): The size of RRoI-Aligning output.
        nms_thr (float): The locality-aware NMS threshold.
        min_width (float): The minimum width of text components.
        max_width (float): The maximum width of text components.
        comp_shrink_ratio (float): The shrink ratio of text components.
        comp_ratio (float): The reciprocal of aspect ratio of text components.
        text_region_thr (float): The threshold for text region probability map.
        center_region_thr (float): The threshold for text center region
            probability map.
        center_region_area_thr (int): The threshold for filtering out
            small-size text center region.
    """

    def __init__(self, k_at_hops, active_connection, node_geo_feat_dim,
                 pooling_scale, pooling_output_size, nms_thr, min_width,
                 max_width, comp_shrink_ratio, comp_ratio, text_region_thr,
                 center_region_thr, center_region_area_thr):

        assert isinstance(k_at_hops, tuple)
        assert isinstance(active_connection, int)
        assert isinstance(node_geo_feat_dim, int)
        assert isinstance(pooling_scale, float)
        assert isinstance(pooling_output_size, tuple)
        assert isinstance(nms_thr, float)
        assert isinstance(min_width, float)
        assert isinstance(max_width, float)
        assert isinstance(comp_shrink_ratio, float)
        assert isinstance(comp_ratio, float)
        assert isinstance(text_region_thr, float)
        assert isinstance(center_region_thr, float)
        assert isinstance(center_region_area_thr, int)

        self.k_at_hops = k_at_hops
        self.active_connection = active_connection
        self.local_graph_depth = len(self.k_at_hops)
        self.node_geo_feat_dim = node_geo_feat_dim
        self.pooling = RROIAlign(pooling_output_size, pooling_scale)
        self.nms_thr = nms_thr
        self.min_width = min_width
        self.max_width = max_width
        self.comp_shrink_ratio = comp_shrink_ratio
        self.comp_ratio = comp_ratio
        self.text_region_thr = text_region_thr
        self.center_region_thr = center_region_thr
        self.center_region_area_thr = center_region_area_thr

    def fill_hole(self, input_mask):
        h, w = input_mask.shape
        canvas = np.zeros((h + 2, w + 2), np.uint8)
        canvas[1:h + 1, 1:w + 1] = input_mask.copy()

        mask = np.zeros((h + 4, w + 4), np.uint8)

        cv2.floodFill(canvas, mask, (0, 0), 1)
        canvas = canvas[1:h + 1, 1:w + 1].astype(np.bool)

        return (~canvas | input_mask.astype(np.uint8))

    def propose_comps(self, top_radius_map, bot_radius_map, sin_map, cos_map,
                      score_map, min_width, max_width, comp_shrink_ratio,
                      comp_ratio):
        """Generate text components.

        Args:
            top_radius_map (ndarray): The predicted distance map from each
                pixel in text center region to top sideline.
            bot_radius_map (ndarray): The predicted distance map from each
                pixel in text center region to bottom sideline.
            sin_map (ndarray): The predicted sin(theta) map.
            cos_map (ndarray): The predicted cos(theta) map.
            score_map (ndarray): The score map for NMS.
            min_width (float): The minimum width of text components.
            max_width (float): The maximum width of text components.
            comp_shrink_ratio (float): The shrink ratio of text components.
            comp_ratio (float): The reciprocal of aspect ratio of text
                components.

        Returns:
            text_comps (ndarray): The text components.
        """

        comp_centers = np.argwhere(score_map > 0)
        comp_centers = comp_centers[np.argsort(comp_centers[:, 0])]
        y = comp_centers[:, 0]
        x = comp_centers[:, 1]

        top_radius = top_radius_map[y, x].reshape((-1, 1)) * comp_shrink_ratio
        bot_radius = bot_radius_map[y, x].reshape((-1, 1)) * comp_shrink_ratio
        sin = sin_map[y, x].reshape((-1, 1))
        cos = cos_map[y, x].reshape((-1, 1))

        top_mid_x_offset = top_radius * cos
        top_mid_y_offset = top_radius * sin
        bot_mid_x_offset = bot_radius * cos
        bot_mid_y_offset = bot_radius * sin

        top_mid_pnt = comp_centers + np.hstack(
            [top_mid_y_offset, top_mid_x_offset])
        bot_mid_pnt = comp_centers - np.hstack(
            [bot_mid_y_offset, bot_mid_x_offset])

        width = (top_radius + bot_radius) * comp_ratio
        width = np.clip(width, min_width, max_width)

        top_left = top_mid_pnt - np.hstack([width * cos, -width * sin
                                            ])[:, ::-1]
        top_right = top_mid_pnt + np.hstack([width * cos, -width * sin
                                             ])[:, ::-1]
        bot_right = bot_mid_pnt + np.hstack([width * cos, -width * sin
                                             ])[:, ::-1]
        bot_left = bot_mid_pnt - np.hstack([width * cos, -width * sin
                                            ])[:, ::-1]

        text_comps = np.hstack([top_left, top_right, bot_right, bot_left])
        score = score_map[y, x].reshape((-1, 1))
        text_comps = np.hstack([text_comps, score])

        return text_comps

    def propose_comps_and_attribs(self, text_region_map, center_region_map,
                                  top_radius_map, bot_radius_map, sin_map,
                                  cos_map):
        """Generate text components and attributes.

        Args:
            text_region_map (ndarray): The predicted text region probability
                map.
            center_region_map (ndarray): The predicted text center region
                probability map.
            top_radius_map (ndarray): The predicted distance map from each
                pixel in text center region to top sideline.
            bot_radius_map (ndarray): The predicted distance map from each
                pixel in text center region to bottom sideline.
            sin_map (ndarray): The predicted sin(theta) map.
            cos_map (ndarray): The predicted cos(theta) map.

        Returns:
            comp_attribs (ndarray): The text components attributes.
            text_comps (ndarray): The text components.
        """

        assert (text_region_map.shape == center_region_map.shape ==
                top_radius_map.shape == bot_radius_map == sin_map.shape ==
                cos_map.shape)
        text_mask = text_region_map > self.text_region_thr
        center_region_mask = (center_region_map >
                              self.center_region_thr) * text_mask

        scale = np.sqrt(1.0 / (sin_map**2 + cos_map**2))
        sin_map, cos_map = sin_map * scale, cos_map * scale

        center_region_mask = self.fill_hole(center_region_mask)
        center_region_contours, _ = cv2.findContours(
            center_region_mask.astype(np.uint8), cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros_like(center_region_mask)
        comp_list = []
        for contour in center_region_contours:
            current_center_mask = mask.copy()
            cv2.drawContours(current_center_mask, [contour], -1, 1, -1)
            if current_center_mask.sum() <= self.center_region_area_thr:
                continue
            score_map = text_region_map * current_center_mask

            text_comp = self.propose_comps(top_radius_map, bot_radius_map,
                                           sin_map, cos_map, score_map,
                                           self.min_width, self.max_width,
                                           self.comp_shrink_ratio,
                                           self.comp_ratio)

            # text_comp = la_nms(text_comp.astype('float32'), self.nms_thr)

            text_comp_mask = mask.copy()
            text_comps_bboxes = text_comp[:, :8].reshape(
                (-1, 4, 2)).astype(np.int32)

            cv2.drawContours(text_comp_mask, text_comps_bboxes, -1, 1, -1)
            if (text_comp_mask * text_mask).sum() < text_comp_mask.sum() * 0.5:
                continue

            comp_list.append(text_comp)

        if len(comp_list) <= 0:
            return None, None

        text_comps = np.vstack(comp_list)

        centers = np.mean(
            text_comps[:, :8].reshape((-1, 4, 2)), axis=1).astype(np.int32)

        x = centers[:, 0]
        y = centers[:, 1]

        h = top_radius_map[y, x].reshape(
            (-1, 1)) + bot_radius_map[y, x].reshape((-1, 1))
        w = np.clip(h * self.comp_ratio, self.min_width, self.max_width)
        sin = sin_map[y, x].reshape((-1, 1))
        cos = cos_map[y, x].reshape((-1, 1))
        x = x.reshape((-1, 1))
        y = y.reshape((-1, 1))
        comp_attribs = np.hstack([x, y, h, w, cos, sin])

        return comp_attribs, text_comps

    def generate_local_graphs(self, sorted_complete_graph, node_feats):
        """Generate local graphs and Graph Convolution Network input data.

        Args:
            sorted_complete_graph (ndarray): The complete graph where nodes are
                sorted according to their Euclidean distance.
            node_feats (tensor): The graph nodes features.

        Returns:
            node_feats_tensor (tensor): The graph nodes features.
            adjacent_matrix_tensor (tensor): The adjacent matrix of graph.
            pivot_inx_tensor (tensor): The pivot indices in local graph.
            knn_inx_tensor (tensor): The k nearest neighbor nodes indexes in
                local graph.
            local_graph_node_tensor (tensor): The indices of nodes in local
                graph.
        """

        assert sorted_complete_graph.ndim == 2
        assert (sorted_complete_graph.shape[0] ==
                sorted_complete_graph.shape[1] == node_feats.shape[0])

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

            local_graph_node_list.append(hops_neighbor_list)
            knn_graph_neighbor_list.append(one_hop_neighbors)

        max_graph_node_num = max([
            len(local_graph_nodes)
            for local_graph_nodes in local_graph_node_list
        ])

        node_normalized_feats = list()
        adjacent_matrix_list = list()
        knn_inx = list()
        pivot_graph_inx = list()
        local_graph_tensor_list = list()

        for graph_inx in range(len(local_graph_node_list)):

            local_graph_nodes = local_graph_node_list[graph_inx]
            local_graph_node_num = len(local_graph_nodes)
            pivot_inx = local_graph_nodes[0]
            knn_graph_neighbors = knn_graph_neighbor_list[graph_inx]
            node_to_graph_inx = {j: i for i, j in enumerate(local_graph_nodes)}

            pivot_node_inx = torch.tensor([
                node_to_graph_inx[pivot_inx],
            ]).type(torch.long)
            knn_inx_in_local_graph = torch.tensor(
                [node_to_graph_inx[i] for i in knn_graph_neighbors],
                dtype=torch.long)
            pivot_feats = node_feats[torch.tensor(pivot_inx, dtype=torch.long)]
            normalized_feats = node_feats[torch.tensor(
                local_graph_nodes, dtype=torch.long)] - pivot_feats

            adjacent_matrix = np.zeros(
                (local_graph_node_num, local_graph_node_num))
            pad_normalized_feats = torch.cat([
                normalized_feats,
                torch.zeros(max_graph_node_num - local_graph_node_num,
                            normalized_feats.shape[1]).to(node_feats.device)
            ],
                                             dim=0)

            for node in local_graph_nodes:
                neighbors = sorted_complete_graph[node,
                                                  1:self.active_connection + 1]
                for neighbor in neighbors:
                    if neighbor in local_graph_nodes:
                        adjacent_matrix[node_to_graph_inx[node],
                                        node_to_graph_inx[neighbor]] = 1
                        adjacent_matrix[node_to_graph_inx[neighbor],
                                        node_to_graph_inx[node]] = 1

            adjacent_matrix = normalize_adjacent_matrix(
                adjacent_matrix, type='DAD')
            adjacent_matrix_tensor = torch.zeros(
                max_graph_node_num, max_graph_node_num).to(node_feats.device)
            adjacent_matrix_tensor[:local_graph_node_num, :
                                   local_graph_node_num] = adjacent_matrix

            local_graph_tensor = torch.tensor(local_graph_nodes)
            local_graph_tensor = torch.cat([
                local_graph_tensor,
                torch.zeros(
                    max_graph_node_num - local_graph_node_num,
                    dtype=torch.long)
            ],
                                           dim=0)

            node_normalized_feats.append(pad_normalized_feats)
            adjacent_matrix_list.append(adjacent_matrix_tensor)
            pivot_graph_inx.append(pivot_node_inx)
            knn_inx.append(knn_inx_in_local_graph)
            local_graph_tensor_list.append(local_graph_tensor)

        node_feats_tensor = torch.stack(node_normalized_feats, 0)
        adjacent_matrix_tensor = torch.stack(adjacent_matrix_list, 0)
        pivot_inx_tensor = torch.stack(pivot_graph_inx, 0)
        knn_inx_tensor = torch.stack(knn_inx, 0)
        local_graph_node_tensor = torch.stack(local_graph_tensor_list, 0)

        return (node_feats_tensor, adjacent_matrix_tensor, pivot_inx_tensor,
                knn_inx_tensor, local_graph_node_tensor)

    def __call__(self, preds, feat_maps):
        """Generate local graphs and Graph Convolution Network input data.

        Args:
            preds (tensor): The predicted maps.
            feat_maps (tensor): The feature maps to extract content features of
                text components.

        Returns:
            node_feats_tensor (tensor): The graph nodes features.
            adjacent_matrix_tensor (tensor): The adjacent matrix of graph.
            pivot_inx_tensor (tensor): The pivot indices in local graph.
            knn_inx_tensor (tensor): The k nearest neighbor nodes indices in
                local graph.
            local_graph_node_tensor (tensor): The indices of nodes in local
                graph.
            text_comps (ndarray): The predicted text components.
        """

        pred_text_region = torch.sigmoid(preds[0, 0]).data.cpu().numpy()
        pred_center_region = torch.sigmoid(preds[0, 1]).data.cpu().numpy()
        pred_sin_map = preds[0, 2].data.cpu().numpy()
        pred_cos_map = preds[0, 3].data.cpu().numpy()
        pred_top_radius_map = preds[0, 4].data.cpu().numpy()
        pred_bot_radius_map = preds[0, 5].data.cpu().numpy()

        comp_attribs, text_comps = self.propose_comps_and_attribs(
            pred_text_region, pred_center_region, pred_top_radius_map,
            pred_bot_radius_map, pred_sin_map, pred_cos_map)

        if comp_attribs is None:
            none_flag = True
            return none_flag, (0, 0, 0, 0, 0, 0)

        comp_centers = comp_attribs[:, 0:2]
        distance_matrix = euclidean_distance_matrix(comp_centers, comp_centers)

        graph_node_geo_feats = embed_geo_feats(comp_attribs,
                                               self.node_geo_feat_dim)
        graph_node_geo_feats = torch.from_numpy(
            graph_node_geo_feats).float().to(preds.device)

        batch_id = np.zeros((comp_attribs.shape[0], 1), dtype=np.float32)
        text_comps_bboxes = np.hstack(
            (batch_id, comp_attribs.astype(np.float32, copy=False)))
        text_comps_bboxes = torch.from_numpy(text_comps_bboxes).float().to(
            preds.device)

        comp_content_feats = self.pooling(feat_maps, text_comps_bboxes)
        comp_content_feats = comp_content_feats.view(
            comp_content_feats.shape[0], -1).to(preds.device)
        node_feats = torch.cat((comp_content_feats, graph_node_geo_feats),
                               dim=-1)

        dist_sort_complete_graph = np.argsort(distance_matrix, axis=1)
        (node_feats_tensor, adjacent_matrix_tensor, pivot_inx_tensor,
         knn_inx_tensor, local_graph_node_tensor) = self.generate_local_graphs(
             dist_sort_complete_graph, node_feats)

        none_flag = False
        return none_flag, (node_feats_tensor, adjacent_matrix_tensor,
                           pivot_inx_tensor, knn_inx_tensor,
                           local_graph_node_tensor, text_comps)

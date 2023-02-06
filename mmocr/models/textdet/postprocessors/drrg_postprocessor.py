# Copyright (c) OpenMMLab. All rights reserved.
import functools
import operator
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
from mmengine.structures import InstanceData
from numpy import ndarray

from mmocr.registry import MODELS
from mmocr.structures import TextDetDataSample
from .base import BaseTextDetPostProcessor


class Node:
    """A simple graph node.

    Args:
        ind (int): The index of the node.
    """

    def __init__(self, ind: int) -> None:
        self.__ind = ind
        self.__links = set()

    @property
    def ind(self) -> int:
        """Current node index."""
        return self.__ind

    @property
    def links(self) -> set:
        """A set of links."""
        return set(self.__links)

    def add_link(self, link_node: 'Node') -> None:
        """Add a link to the node.

        Args:
            link_node (Node): The link node.
        """
        self.__links.add(link_node)
        link_node.__links.add(self)


@MODELS.register_module()
class DRRGPostprocessor(BaseTextDetPostProcessor):
    """Merge text components and construct boundaries of text instances.

    Args:
        link_thr (float): The edge score threshold. Defaults to 0.8.
        edge_len_thr (int or float): The edge length threshold. Defaults to 50.
        rescale_fields (list[str]): The bbox/polygon field names to
            be rescaled. If None, no rescaling will be performed. Defaults to
            [polygons'].
    """

    def __init__(self,
                 link_thr: float = 0.8,
                 edge_len_thr: Union[int, float] = 50.,
                 rescale_fields=['polygons'],
                 **kwargs) -> None:
        super().__init__(rescale_fields=rescale_fields)
        assert isinstance(link_thr, float)
        assert isinstance(edge_len_thr, (int, float))
        self.link_thr = link_thr
        self.edge_len_thr = edge_len_thr

    def get_text_instances(self, pred_results: Tuple[ndarray, ndarray,
                                                     ndarray],
                           data_sample: TextDetDataSample
                           ) -> TextDetDataSample:
        """Get text instance predictions of one image.

        Args:
            pred_result (tuple(ndarray, ndarray, ndarray)): Prediction results
                edge, score and text_comps. Each of shape
                :math:`(N_{edges}, 2)`, :math:`(N_{edges},)` and
                :math:`(M, 9)`, respectively.
            data_sample (TextDetDataSample): Datasample of an image.

        Returns:
            TextDetDataSample: The original dataSample with predictions filled
            in. Polygons and results are saved in
            ``TextDetDataSample.pred_instances.polygons``. The confidence
            scores are saved in ``TextDetDataSample.pred_instances.scores``.
        """

        data_sample.pred_instances = InstanceData()
        polys = []
        scores = []

        pred_edges, pred_scores, text_comps = pred_results

        if pred_edges is not None:
            assert len(pred_edges) == len(pred_scores)
            assert text_comps.ndim == 2
            assert text_comps.shape[1] == 9

            vertices, score_dict = self._graph_propagation(
                pred_edges, pred_scores, text_comps)
            clusters = self._connected_components(vertices, score_dict)
            pred_labels = self._clusters2labels(clusters, text_comps.shape[0])
            text_comps, pred_labels = self._remove_single(
                text_comps, pred_labels)
            polys, scores = self._comps2polys(text_comps, pred_labels)

        data_sample.pred_instances.polygons = polys
        data_sample.pred_instances.scores = torch.FloatTensor(scores)

        return data_sample

    def split_results(self, pred_results: Tuple[ndarray, ndarray,
                                                ndarray]) -> List[Tuple]:
        """Split batched elements in pred_results along the first dimension
        into ``batch_num`` sub-elements and regather them into a list of dicts.

        However, DRRG only outputs one batch at inference time, so this
        function is a no-op.
        """
        return [pred_results]

    def _graph_propagation(self, edges: ndarray, scores: ndarray,
                           text_comps: ndarray) -> Tuple[List[Node], Dict]:
        """Propagate edge score information and construct graph. This code was
        partially adapted from https://github.com/GXYM/DRRG licensed under the
        MIT license.

        Args:
            edges (ndarray): The edge array of shape N * 2, each row is a node
                index pair that makes up an edge in graph.
            scores (ndarray): The edge score array.
            text_comps (ndarray): The text components.

        Returns:
            tuple(vertices, score_dict):

            - vertices (list[Node]): The Nodes in graph.
            - score_dict (dict): The edge score dict.
        """
        assert edges.ndim == 2
        assert edges.shape[1] == 2
        assert edges.shape[0] == scores.shape[0]
        assert text_comps.ndim == 2

        edges = np.sort(edges, axis=1)
        score_dict = {}
        for i, edge in enumerate(edges):
            if text_comps is not None:
                box1 = text_comps[edge[0], :8].reshape(4, 2)
                box2 = text_comps[edge[1], :8].reshape(4, 2)
                center1 = np.mean(box1, axis=0)
                center2 = np.mean(box2, axis=0)
                distance = np.linalg.norm(center1 - center2)
                if distance > self.edge_len_thr:
                    scores[i] = 0
            if (edge[0], edge[1]) in score_dict:
                score_dict[edge[0], edge[1]] = 0.5 * (
                    score_dict[edge[0], edge[1]] + scores[i])
            else:
                score_dict[edge[0], edge[1]] = scores[i]

        nodes = np.sort(np.unique(edges.flatten()))
        mapping = -1 * np.ones((np.max(nodes) + 1), dtype=int)
        mapping[nodes] = np.arange(nodes.shape[0])
        order_inds = mapping[edges]
        vertices = [Node(node) for node in nodes]
        for ind in order_inds:
            vertices[ind[0]].add_link(vertices[ind[1]])

        return vertices, score_dict

    def _connected_components(self, nodes: List[Node],
                              score_dict: Dict) -> List[List[Node]]:
        """Conventional connected components searching. This code was partially
        adapted from https://github.com/GXYM/DRRG licensed under the MIT
        license.

        Args:
            nodes (list[Node]): The list of Node objects.
            score_dict (dict): The edge score dict.

        Returns:
            List[list[Node]]: The clustered Node objects.
        """
        assert isinstance(nodes, list)
        assert all([isinstance(node, Node) for node in nodes])
        assert isinstance(score_dict, dict)

        clusters = []
        nodes = set(nodes)
        while nodes:
            node = nodes.pop()
            cluster = {node}
            node_queue = [node]
            while node_queue:
                node = node_queue.pop(0)
                neighbors = {
                    neighbor
                    for neighbor in node.links if score_dict[tuple(
                        sorted([node.ind, neighbor.ind]))] >= self.link_thr
                }
                neighbors.difference_update(cluster)
                nodes.difference_update(neighbors)
                cluster.update(neighbors)
                node_queue.extend(neighbors)
            clusters.append(list(cluster))
        return clusters

    def _clusters2labels(self, clusters: List[List[Node]],
                         num_nodes: int) -> ndarray:
        """Convert clusters of Node to text component labels. This code was
        partially adapted from https://github.com/GXYM/DRRG licensed under the
        MIT license.

        Args:
            clusters (List[list[Node]]): The clusters of Node objects.
            num_nodes (int): The total node number of graphs in an image.

        Returns:
            ndarray: The node label array.
        """
        assert isinstance(clusters, list)
        assert all([isinstance(cluster, list) for cluster in clusters])
        assert all([
            isinstance(node, Node) for cluster in clusters for node in cluster
        ])
        assert isinstance(num_nodes, int)

        node_labels = np.zeros(num_nodes)
        for cluster_ind, cluster in enumerate(clusters):
            for node in cluster:
                node_labels[node.ind] = cluster_ind
        return node_labels

    def _remove_single(self, text_comps: ndarray,
                       comp_pred_labels: ndarray) -> Tuple[ndarray, ndarray]:
        """Remove isolated text components. This code was partially adapted
        from https://github.com/GXYM/DRRG licensed under the MIT license.

        Args:
            text_comps (ndarray): The text components.
            comp_pred_labels (ndarray): The clustering labels of text
                components.

        Returns:
            tuple(filtered_text_comps, comp_pred_labels):

            - filtered_text_comps (ndarray): The text components with isolated
              ones removed.
            - comp_pred_labels (ndarray): The clustering labels with labels of
              isolated text components removed.
        """
        assert text_comps.ndim == 2
        assert text_comps.shape[0] == comp_pred_labels.shape[0]

        single_flags = np.zeros_like(comp_pred_labels)
        pred_labels = np.unique(comp_pred_labels)
        for label in pred_labels:
            current_label_flag = (comp_pred_labels == label)
            if np.sum(current_label_flag) == 1:
                single_flags[np.where(current_label_flag)[0][0]] = 1
        keep_ind = [
            i for i in range(len(comp_pred_labels)) if not single_flags[i]
        ]
        filtered_text_comps = text_comps[keep_ind, :]
        filtered_labels = comp_pred_labels[keep_ind]

        return filtered_text_comps, filtered_labels

    def _comps2polys(self, text_comps: ndarray, comp_pred_labels: ndarray
                     ) -> Tuple[List[ndarray], List[float]]:
        """Construct text instance boundaries from clustered text components.
        This code was partially adapted from https://github.com/GXYM/DRRG
        licensed under the MIT license.

        Args:
            text_comps (ndarray): The text components.
            comp_pred_labels (ndarray): The clustering labels of text
                components.

        Returns:
            tuple(boundaries, scores):

            - boundaries (list[ndarray]): The predicted boundaries of text
              instances.
            - scores (list[float]): The boundary scores.
        """
        assert text_comps.ndim == 2
        assert len(text_comps) == len(comp_pred_labels)
        boundaries = []
        scores = []
        if len(text_comps) < 1:
            return boundaries, scores
        for cluster_ind in range(0, int(np.max(comp_pred_labels)) + 1):
            cluster_comp_inds = np.where(comp_pred_labels == cluster_ind)
            text_comp_boxes = text_comps[cluster_comp_inds, :8].reshape(
                (-1, 4, 2)).astype(np.int32)
            score = np.mean(text_comps[cluster_comp_inds, -1])

            if text_comp_boxes.shape[0] < 1:
                continue

            elif text_comp_boxes.shape[0] > 1:
                centers = np.mean(
                    text_comp_boxes, axis=1).astype(np.int32).tolist()
                shortest_path = self._min_connect_path(centers)
                text_comp_boxes = text_comp_boxes[shortest_path]
                top_line = np.mean(
                    text_comp_boxes[:, 0:2, :],
                    axis=1).astype(np.int32).tolist()
                bot_line = np.mean(
                    text_comp_boxes[:, 2:4, :],
                    axis=1).astype(np.int32).tolist()
                top_line, bot_line = self._fix_corner(top_line, bot_line,
                                                      text_comp_boxes[0],
                                                      text_comp_boxes[-1])
                boundary_points = top_line + bot_line[::-1]

            else:
                top_line = text_comp_boxes[0, 0:2, :].astype(np.int32).tolist()
                bot_line = text_comp_boxes[0, 2:4:-1, :].astype(
                    np.int32).tolist()
                boundary_points = top_line + bot_line

            boundary = [p for coord in boundary_points for p in coord]
            boundaries.append(np.array(boundary, dtype=np.float32))
            scores.append(score)

        return boundaries, scores

    def _norm2(self, point1: List[int], point2: List[int]) -> float:
        """Calculate the norm of two points."""
        return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

    def _min_connect_path(self, points: List[List[int]]) -> List[List[int]]:
        """Find the shortest path to traverse all points. This code was
        partially adapted from https://github.com/GXYM/DRRG licensed under the
        MIT license.

        Args:
            points(List[list[int]]): The point sequence
                [[x0, y0], [x1, y1], ...].

        Returns:
            List[list[int]]: The shortest index path.
        """
        assert isinstance(points, list)
        assert all([isinstance(point, list) for point in points])
        assert all(
            [isinstance(coord, int) for point in points for coord in point])

        points_queue = points.copy()
        shortest_path = []
        current_edge = [[], []]

        edge_dict0 = {}
        edge_dict1 = {}
        current_edge[0] = points_queue[0]
        current_edge[1] = points_queue[0]
        points_queue.remove(points_queue[0])
        while points_queue:
            for point in points_queue:
                length0 = self._norm2(point, current_edge[0])
                edge_dict0[length0] = [point, current_edge[0]]
                length1 = self._norm2(current_edge[1], point)
                edge_dict1[length1] = [current_edge[1], point]
            key0 = min(edge_dict0.keys())
            key1 = min(edge_dict1.keys())

            if key0 <= key1:
                start = edge_dict0[key0][0]
                end = edge_dict0[key0][1]
                shortest_path.insert(0,
                                     [points.index(start),
                                      points.index(end)])
                points_queue.remove(start)
                current_edge[0] = start
            else:
                start = edge_dict1[key1][0]
                end = edge_dict1[key1][1]
                shortest_path.append([points.index(start), points.index(end)])
                points_queue.remove(end)
                current_edge[1] = end

            edge_dict0 = {}
            edge_dict1 = {}

        shortest_path = functools.reduce(operator.concat, shortest_path)
        shortest_path = sorted(set(shortest_path), key=shortest_path.index)

        return shortest_path

    def _in_contour(self, contour: ndarray, point: ndarray) -> bool:
        """Whether a point is in a contour."""
        x, y = point
        return cv2.pointPolygonTest(contour, (int(x), int(y)), False) > 0.5

    def _fix_corner(self, top_line: List[List[int]], btm_line: List[List[int]],
                    start_box: ndarray, end_box: ndarray
                    ) -> Tuple[List[List[int]], List[List[int]]]:
        """Add corner points to predicted side lines. This code was partially
        adapted from https://github.com/GXYM/DRRG licensed under the MIT
        license.

        Args:
            top_line (List[list[int]]): The predicted top sidelines of text
                instance.
            btm_line (List[list[int]]): The predicted bottom sidelines of text
                instance.
            start_box (ndarray): The first text component box.
            end_box (ndarray): The last text component box.

        Returns:
            tuple(top_line, bot_line):

            - top_line (List[list[int]]): The top sidelines with corner point
              added.
            - bot_line (List[list[int]]): The bottom sidelines with corner
              point added.
        """
        assert isinstance(top_line, list)
        assert all(isinstance(point, list) for point in top_line)
        assert isinstance(btm_line, list)
        assert all(isinstance(point, list) for point in btm_line)
        assert start_box.shape == end_box.shape == (4, 2)

        contour = np.array(top_line + btm_line[::-1])
        start_left_mid = (start_box[0] + start_box[3]) / 2
        start_right_mid = (start_box[1] + start_box[2]) / 2
        end_left_mid = (end_box[0] + end_box[3]) / 2
        end_right_mid = (end_box[1] + end_box[2]) / 2
        if not self._in_contour(contour, start_left_mid):
            top_line.insert(0, start_box[0].tolist())
            btm_line.insert(0, start_box[3].tolist())
        elif not self._in_contour(contour, start_right_mid):
            top_line.insert(0, start_box[1].tolist())
            btm_line.insert(0, start_box[2].tolist())
        if not self._in_contour(contour, end_left_mid):
            top_line.append(end_box[0].tolist())
            btm_line.append(end_box[3].tolist())
        elif not self._in_contour(contour, end_right_mid):
            top_line.append(end_box[1].tolist())
            btm_line.append(end_box[2].tolist())
        return top_line, btm_line

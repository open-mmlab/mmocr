import functools
import operator

import cv2
import numpy as np
import torch
from numpy.linalg import norm


def normalize_adjacent_matrix(A, mode='AD'):
    """Normalize adjacent matrix for GCN. This code was partially adapted from
    https://github.com/GXYM/DRRG.

    Args:
        A (ndarray): The adjacent matrix.
        mode (string): The normalize mode.

    returns:
        G (ndarray): The normalized adjacent matrix.
    """
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]
    assert mode in ['DAD', 'AD']
    if mode == 'DAD':
        A = A + np.eye(A.shape[0])
        d = np.sum(A, axis=0)
        d_inv = np.power(d, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_inv = np.diag(d_inv)
        G = A.dot(d_inv).transpose().dot(d_inv)
        G = torch.from_numpy(G)
    elif mode == 'AD':
        A = A + np.eye(A.shape[0])
        A = torch.from_numpy(A)
        D = A.sum(1, keepdim=True)
        G = A.div(D)
    else:
        raise NotImplementedError
    return G


def euclidean_distance_matrix(A, B):
    """Calculate euclidean distance matrix.

    Args:
        A (ndarray): The point sequence.
        B (ndarray): The point sequence in the same dimension as A.

    returns:
        D (ndarray): The euclidean distance matrix.
    """
    assert A.ndim == 2
    assert B.ndim == 2
    assert A.shape[1] == B.shape[1]

    m = A.shape[0]
    n = B.shape[0]

    A_dots = (A * A).sum(axis=1).reshape((m, 1)) * np.ones(shape=(1, n))
    B_dots = (B * B).sum(axis=1) * np.ones(shape=(m, 1))
    D_squared = A_dots + B_dots - 2 * A.dot(B.T)

    zero_mask = np.less(D_squared, 0.0)
    D_squared[zero_mask] = 0.0
    D = np.sqrt(D_squared)
    return D


def embed_geo_feats(geo_feats, out_feat_len):
    """Embed geometric feature of text components. This code was partially
    adapted from https://github.com/GXYM/DRRG.

    Args:
        geo_feats (ndarray): The geometric features of text components.
        out_feat_len (int): The length of output feature vector.

    Returns:
        embedded_feats (ndarray): The embedded geometric features.
    """
    assert isinstance(out_feat_len, int)
    assert out_feat_len >= geo_feats.shape[1]

    comp_num = geo_feats.shape[0]
    feat_dim = geo_feats.shape[1]
    feat_repeat_times = out_feat_len // feat_dim
    residue_dim = out_feat_len % feat_dim

    if residue_dim > 0:
        embed_wave = np.array([
            np.power(1000, 2.0 * (j // 2) / feat_repeat_times + 1)
            for j in range(feat_repeat_times + 1)
        ]).reshape((feat_repeat_times + 1, 1, 1))
        repeat_feats = np.repeat(
            np.expand_dims(geo_feats, axis=0), feat_repeat_times, axis=0)
        residue_feats = np.hstack([
            geo_feats[:, 0:residue_dim],
            np.zeros((comp_num, feat_dim - residue_dim))
        ])
        repeat_feats = np.stack([repeat_feats, residue_feats], axis=0)
        embedded_feats = repeat_feats / embed_wave
        embedded_feats[:, 0::2] = np.sin(embedded_feats[:, 0::2])
        embedded_feats[:, 1::2] = np.cos(embedded_feats[:, 1::2])
        embedded_feats = np.transpose(embedded_feats, (1, 0, 2)).reshape(
            (comp_num, -1))[:, 0:out_feat_len]
    else:
        embed_wave = np.array([
            np.power(1000, 2.0 * (j // 2) / feat_repeat_times)
            for j in range(feat_repeat_times)
        ]).reshape((feat_repeat_times, 1, 1))
        repeat_feats = np.repeat(
            np.expand_dims(geo_feats, axis=0), feat_repeat_times, axis=0)
        embedded_feats = repeat_feats / embed_wave
        embedded_feats[:, 0::2] = np.sin(embedded_feats[:, 0::2])
        embedded_feats[:, 1::2] = np.cos(embedded_feats[:, 1::2])
        embedded_feats = np.transpose(embedded_feats, (1, 0, 2)).reshape(
            (comp_num, -1)).astype(np.float32)

    return embedded_feats


class Node(object):

    def __init__(self, ind):
        self.__ind = ind
        self.__links = set()

    @property
    def ind(self):
        return self.__ind

    @property
    def links(self):
        return set(self.__links)

    def add_link(self, link_node):
        self.__links.add(link_node)
        link_node.__links.add(self)


def graph_propagation(edges, scores, text_comps, edge_len_thr=50.):
    """Propagate edge score information and construct graph. This code was
    partially adapted from https://github.com/GXYM/DRRG.

    Args:
        edges (ndarray): The edge array of shape N * 2, each row is a node
            index pair that make up an edge in graph.
        scores (ndarray): The edge score array.
        text_comps (ndarray): The text components.
        edge_len_thr (float): The edge length threshold.

    Returns:
        vertices (list[Node]): The Nodes in graph.
        score_dict (dict): The edge score dict.
    """
    assert edges.ndim == 2
    assert edges.shape[1] == 2
    assert edges.shape[0] == scores.shape[0]
    assert text_comps.ndim == 2
    assert isinstance(edge_len_thr, float)

    edges = np.sort(edges, axis=1)
    score_dict = {}
    for i, edge in enumerate(edges):
        if text_comps is not None:
            box1 = text_comps[edge[0], :8].reshape(4, 2)
            box2 = text_comps[edge[1], :8].reshape(4, 2)
            center1 = np.mean(box1, axis=0)
            center2 = np.mean(box2, axis=0)
            distance = norm(center1 - center2)
            if distance > edge_len_thr:
                scores[i] = 0
        if (edge[0], edge[1]) in score_dict:
            score_dict[edge[0], edge[1]] = 0.5 * (
                score_dict[edge[0], edge[1]] + scores[i])
        else:
            score_dict[edge[0], edge[1]] = scores[i]

    nodes = np.sort(np.unique(edges.flatten()))
    mapping = -1 * np.ones((np.max(nodes) + 1), dtype=np.int)
    mapping[nodes] = np.arange(nodes.shape[0])
    order_inds = mapping[edges]
    vertices = [Node(node) for node in nodes]
    for ind in order_inds:
        vertices[ind[0]].add_link(vertices[ind[1]])

    return vertices, score_dict


def connected_components(nodes, score_dict, link_thr):
    """Conventional connected components searching. This code was partially
    adapted from https://github.com/GXYM/DRRG.

    Args:
        nodes (list[Node]): The list of Node object.
        score_dict (dict): The edge score dict.
        link_thr (float): The link threshold.

    Returns:
        clusters (List[list[Node]]): The clustered Node objects.
    """
    assert isinstance(nodes, list)
    assert all([isinstance(node, Node) for node in nodes])
    assert isinstance(score_dict, dict)
    assert isinstance(link_thr, float)

    clusters = []
    nodes = set(nodes)
    while nodes:
        node = nodes.pop()
        cluster = {node}
        node_queue = [node]
        while node_queue:
            node = node_queue.pop(0)
            neighbors = set([
                neighbor for neighbor in node.links if
                score_dict[tuple(sorted([node.ind, neighbor.ind]))] >= link_thr
            ])
            neighbors.difference_update(cluster)
            nodes.difference_update(neighbors)
            cluster.update(neighbors)
            node_queue.extend(neighbors)
        clusters.append(list(cluster))
    return clusters


def clusters2labels(clusters, node_num):
    """Convert clusters of Node to text component labels. This code was
    partially adapted from https://github.com/GXYM/DRRG.

    Args:
        clusters (List[list[Node]]): The clusters of Node object.
        node_num (int): The total node number of graphs in an image.

    Returns:
        node_labels (ndarray): The node label array.
    """
    assert isinstance(clusters, list)
    assert all([isinstance(cluster, list) for cluster in clusters])
    assert all(
        [isinstance(node, Node) for cluster in clusters for node in cluster])
    assert isinstance(node_num, int)

    node_labels = np.zeros(node_num)
    for cluster_ind, cluster in enumerate(clusters):
        for node in cluster:
            node_labels[node.ind] = cluster_ind
    return node_labels


def remove_single(text_comps, comp_pred_labels):
    """Remove isolated text components. This code was partially adapted from
    https://github.com/GXYM/DRRG.

    Args:
        text_comps (ndarray): The text components.
        comp_pred_labels (ndarray): The clustering label of text components.

    Returns:
        filtered_text_comps (ndarray): The text components with isolated ones
            removed.
        comp_pred_labels (ndarray): The clustering labels with labels of
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
    keep_ind = [i for i in range(len(comp_pred_labels)) if not single_flags[i]]
    filtered_text_comps = text_comps[keep_ind, :]
    filtered_labels = comp_pred_labels[keep_ind]

    return filtered_text_comps, filtered_labels


def norm2(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5


def min_connect_path(points):
    """Find the shortest path to traverse all points. This code was partially
    adapted from https://github.com/GXYM/DRRG.

    Args:
        points(List[list[int]]): The point sequence [[x0, y0], [x1, y1], ...].

    Returns:
        shortest_path(List[list[int]]): The shortest index path.
    """
    assert isinstance(points, list)
    assert all([isinstance(point, list) for point in points])
    assert all([isinstance(coord, int) for point in points for coord in point])

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
            length0 = norm2(point, current_edge[0])
            edge_dict0[length0] = [point, current_edge[0]]
            length1 = norm2(current_edge[1], point)
            edge_dict1[length1] = [current_edge[1], point]
        key0 = min(edge_dict0.keys())
        key1 = min(edge_dict1.keys())

        if key0 <= key1:
            start = edge_dict0[key0][0]
            end = edge_dict0[key0][1]
            shortest_path.insert(0, [points.index(start), points.index(end)])
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


def in_contour(cont, point):
    x, y = point
    is_inner = cv2.pointPolygonTest(cont, (int(x), int(y)), False) > 0.5
    return is_inner


def fix_corner(top_line, bot_line, start_box, end_box):
    """Add corner points in predicted sidelines. This code was partially
    adapted from https://github.com/GXYM/DRRG.

    Args:
        top_line (List[list[int]]): The predicted top sideline of text
            instance.
        bot_line (List[list[int]]): The predicted bottom sideline of text
            instance.
        start_box (ndarray): The first text component box.
        end_box (ndarray): The last text component box.

    Returns:
        top_line (List[list[int]]): The top sideline with corner point added.
        bot_line (List[list[int]]): The bottom sideline with corner point
            added.
    """
    assert isinstance(top_line, list)
    assert all(isinstance(point, list) for point in top_line)
    assert isinstance(bot_line, list)
    assert all(isinstance(point, list) for point in bot_line)
    assert start_box.shape == end_box.shape == (4, 2)

    contour = np.array(top_line + bot_line[::-1])
    start_left_mid = (start_box[0] + start_box[3]) / 2
    start_right_mid = (start_box[1] + start_box[2]) / 2
    end_left_mid = (end_box[0] + end_box[3]) / 2
    end_right_mid = (end_box[1] + end_box[2]) / 2
    if not in_contour(contour, start_left_mid):
        top_line.insert(0, start_box[0].tolist())
        bot_line.insert(0, start_box[3].tolist())
    elif not in_contour(contour, start_right_mid):
        top_line.insert(0, start_box[1].tolist())
        bot_line.insert(0, start_box[2].tolist())
    if not in_contour(contour, end_left_mid):
        top_line.append(end_box[0].tolist())
        bot_line.append(end_box[3].tolist())
    elif not in_contour(contour, end_right_mid):
        top_line.append(end_box[1].tolist())
        bot_line.append(end_box[2].tolist())
    return top_line, bot_line


def comps2boundaries(text_comps, comp_pred_labels):
    """Construct text instance boundaries from clustered text components. This
    code was partially adapted from https://github.com/GXYM/DRRG.

    Args:
        text_comps (ndarray): The text components.
        comp_pred_labels (ndarray): The clustering label of text components.

    Returns:
        boundaries (List[list[float]]): The predicted boundaries of text
            instances.
    """
    assert text_comps.ndim == 2
    assert len(text_comps) == len(comp_pred_labels)
    boundaries = []
    if len(text_comps) < 1:
        return boundaries
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
            shortest_path = min_connect_path(centers)
            text_comp_boxes = text_comp_boxes[shortest_path]
            top_line = np.mean(
                text_comp_boxes[:, 0:2, :], axis=1).astype(np.int32).tolist()
            bot_line = np.mean(
                text_comp_boxes[:, 2:4, :], axis=1).astype(np.int32).tolist()
            top_line, bot_line = fix_corner(top_line, bot_line,
                                            text_comp_boxes[0],
                                            text_comp_boxes[-1])
            boundary_points = top_line + bot_line[::-1]

        else:
            top_line = text_comp_boxes[0, 0:2, :].astype(np.int32).tolist()
            bot_line = text_comp_boxes[0, 2:4:-1, :].astype(np.int32).tolist()
            boundary_points = top_line + bot_line

        boundary = [p for coord in boundary_points for p in coord] + [score]
        boundaries.append(boundary)

    return boundaries


def merge_text_comps(edges, scores, text_comps, link_thr):
    """Merge text components and construct boundaries of text instances.

    Args:
        edges (ndarray): The edge array of shape N * 2, each row is a node
            index pair that make up an edge in graph.
        scores (ndarray): The edge score array.
        text_comps (ndarray): The text components.
        link_thr (float): The edge score threshold.

    Returns:
        boundaries (List[list[float]]): The predicted boundaries of text
            instances.
    """
    assert len(edges) == len(scores)
    assert text_comps.ndim == 2
    assert text_comps.shape[1] == 9
    assert isinstance(link_thr, float)
    vertices, score_dict = graph_propagation(edges, scores, text_comps)
    clusters = connected_components(vertices, score_dict, link_thr)
    pred_labels = clusters2labels(clusters, text_comps.shape[0])
    text_comps, pred_labels = remove_single(text_comps, pred_labels)
    boundaries = comps2boundaries(text_comps, pred_labels)

    return boundaries

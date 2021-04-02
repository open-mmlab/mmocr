import functools
import operator
from typing import List

import cv2
import numpy as np
import torch
from numpy.linalg import norm


def normalize_adjacent_matrix(A, type='AD'):
    """Normalize adjacent matrix for GCN.

    This was from repo https://github.com/GXYM/DRRG.
    """
    if type == 'DAD':
        # d is  Degree of nodes A=A+I
        # L = D^-1/2 A D^-1/2
        A = A + np.eye(A.shape[0])  # A=A+I
        d = np.sum(A, axis=0)
        d_inv = np.power(d, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_inv = np.diag(d_inv)
        G = A.dot(d_inv).transpose().dot(d_inv)
        G = torch.from_numpy(G)
    elif type == 'AD':
        A = A + np.eye(A.shape[0])  # A=A+I
        A = torch.from_numpy(A)
        D = A.sum(1, keepdim=True)
        G = A.div(D)
    else:
        A = A + np.eye(A.shape[0])  # A=A+I
        A = torch.from_numpy(A)
        D = A.sum(1, keepdim=True)
        D = np.diag(D)
        G = D - A
    return G


def euclidean_distance_matrix(A, B):
    """Calculate the Euclidean distance matrix."""

    M = A.shape[0]
    N = B.shape[0]

    assert A.shape[1] == B.shape[1]

    A_dots = (A * A).sum(axis=1).reshape((M, 1)) * np.ones(shape=(1, N))
    B_dots = (B * B).sum(axis=1) * np.ones(shape=(M, 1))
    D_squared = A_dots + B_dots - 2 * A.dot(B.T)

    zero_mask = np.less(D_squared, 0.0)
    D_squared[zero_mask] = 0.0
    return np.sqrt(D_squared)


def embed_geo_feats(geo_feats, out_dim):
    """Embed geometric features of text components. This was partially adapted
    from https://github.com/GXYM/DRRG.

    Args:
        geo_feats (ndarray): The geometric features of text components.
        out_dim (int): The output dimension.

    Returns:
        embedded_feats (ndarray): The embedded geometric features.
    """
    assert isinstance(out_dim, int)
    assert out_dim >= geo_feats.shape[1]
    comp_num = geo_feats.shape[0]
    feat_dim = geo_feats.shape[1]
    feat_repeat_times = out_dim // feat_dim
    residue_dim = out_dim % feat_dim

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
            (comp_num, -1))[:, 0:out_dim]
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
            (comp_num, -1))

    return embedded_feats


def min_connect_path(list_all: List[list]):
    """This is from https://github.com/GXYM/DRRG."""

    list_nodo = list_all.copy()
    res: List[List[int]] = []
    ept = [0, 0]

    def norm2(a, b):
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

    dict00 = {}
    dict11 = {}
    ept[0] = list_nodo[0]
    ept[1] = list_nodo[0]
    list_nodo.remove(list_nodo[0])
    while list_nodo:
        for i in list_nodo:
            length0 = norm2(i, ept[0])
            dict00[length0] = [i, ept[0]]
            length1 = norm2(ept[1], i)
            dict11[length1] = [ept[1], i]
        key0 = min(dict00.keys())
        key1 = min(dict11.keys())

        if key0 <= key1:
            ss = dict00[key0][0]
            ee = dict00[key0][1]
            res.insert(0, [list_all.index(ss), list_all.index(ee)])
            list_nodo.remove(ss)
            ept[0] = ss
        else:
            ss = dict11[key1][0]
            ee = dict11[key1][1]
            res.append([list_all.index(ss), list_all.index(ee)])
            list_nodo.remove(ee)
            ept[1] = ee

        dict00 = {}
        dict11 = {}

    path = functools.reduce(operator.concat, res)
    path = sorted(set(path), key=path.index)

    return res, path


def clusters2labels(clusters, node_num):
    """This is from https://github.com/GXYM/DRRG."""
    labels = (-1) * np.ones((node_num, ))
    for cluster_inx, cluster in enumerate(clusters):
        for node in cluster:
            labels[node.inx] = cluster_inx
    assert np.sum(labels < 0) < 1
    return labels


def remove_single(text_comps, pred):
    """Remove isolated single text components.

    This is from https://github.com/GXYM/DRRG.
    """
    single_flags = np.zeros_like(pred)
    pred_labels = np.unique(pred)
    for label in pred_labels:
        current_label_flag = pred == label
        if np.sum(current_label_flag) == 1:
            single_flags[np.where(current_label_flag)[0][0]] = 1
    remain_inx = [i for i in range(len(pred)) if not single_flags[i]]
    remain_inx = np.asarray(remain_inx)
    return text_comps[remain_inx, :], pred[remain_inx]


class Node:

    def __init__(self, inx):
        self.__inx = inx
        self.__links = set()

    @property
    def inx(self):
        return self.__inx

    @property
    def links(self):
        return set(self.__links)

    def add_link(self, other, score):
        self.__links.add(other)
        other.__links.add(self)


def connected_components(nodes, score_dict, thr):
    """Connected components searching.

    This is from https://github.com/GXYM/DRRG.
    """

    result = []
    nodes = set(nodes)
    while nodes:
        node = nodes.pop()
        group = {node}
        queue = [node]
        while queue:
            node = queue.pop(0)
            if thr is not None:
                neighbors = {
                    linked_neighbor
                    for linked_neighbor in node.links if score_dict[tuple(
                        sorted([node.inx, linked_neighbor.inx]))] >= thr
                }
            else:
                neighbors = node.links
            neighbors.difference_update(group)
            nodes.difference_update(neighbors)
            group.update(neighbors)
            queue.extend(neighbors)
        result.append(group)
    return result


def graph_propagation(edges,
                      scores,
                      link_thr,
                      bboxes=None,
                      dis_thr=50,
                      pool='avg'):
    """Propagate graph linkage score information.

    This is from repo https://github.com/GXYM/DRRG.
    """
    edges = np.sort(edges, axis=1)

    score_dict = {}
    if pool is None:
        for i, edge in enumerate(edges):
            score_dict[edge[0], edge[1]] = scores[i]
    elif pool == 'avg':
        for i, edge in enumerate(edges):
            if bboxes is not None:
                box1 = bboxes[edge[0]][:8].reshape(4, 2)
                box2 = bboxes[edge[1]][:8].reshape(4, 2)
                center1 = np.mean(box1, axis=0)
                center2 = np.mean(box2, axis=0)
                dst = norm(center1 - center2)
                if dst > dis_thr:
                    scores[i] = 0
            if (edge[0], edge[1]) in score_dict:
                score_dict[edge[0], edge[1]] = 0.5 * (
                    score_dict[edge[0], edge[1]] + scores[i])
            else:
                score_dict[edge[0], edge[1]] = scores[i]

    elif pool == 'max':
        for i, edge in enumerate(edges):
            if (edge[0], edge[1]) in score_dict:
                score_dict[edge[0],
                           edge[1]] = max(score_dict[edge[0], edge[1]],
                                          scores[i])
            else:
                score_dict[edge[0], edge[1]] = scores[i]
    else:
        raise ValueError('Pooling operation not supported')

    nodes = np.sort(np.unique(edges.flatten()))
    mapping = -1 * np.ones((nodes.max() + 1), dtype=np.int)
    mapping[nodes] = np.arange(nodes.shape[0])
    link_inx = mapping[edges]
    vertex = [Node(node) for node in nodes]
    for link, score in zip(link_inx, scores):
        vertex[link[0]].add_link(vertex[link[1]], score)

    clusters = connected_components(vertex, score_dict, link_thr)

    return clusters


def in_contour(cont, point):
    x, y = point
    return cv2.pointPolygonTest(cont, (x, y), False) > 0


def select_edge(cont, box):
    """This is from repo https://github.com/GXYM/DRRG."""
    cont = np.array(cont)
    box = box.astype(np.int32)
    c1 = np.array(0.5 * (box[0, :] + box[3, :]), dtype=np.int)
    c2 = np.array(0.5 * (box[1, :] + box[2, :]), dtype=np.int)

    if not in_contour(cont, c1):
        return [box[0, :].tolist(), box[3, :].tolist()]
    elif not in_contour(cont, c2):
        return [box[1, :].tolist(), box[2, :].tolist()]
    else:
        return None


def comps2boundary(text_comps, final_pred):
    """Propose text components and generate local graphs.

    This is from repo https://github.com/GXYM/DRRG.
    """
    bbox_contours = list()
    for inx in range(0, int(np.max(final_pred)) + 1):
        current_instance = np.where(final_pred == inx)
        boxes = text_comps[current_instance, :8].reshape(
            (-1, 4, 2)).astype(np.int32)

        boundary_point = None
        if boxes.shape[0] > 1:
            centers = np.mean(boxes, axis=1).astype(np.int32).tolist()
            paths, routes_path = min_connect_path(centers)
            boxes = boxes[routes_path]
            top = np.mean(boxes[:, 0:2, :], axis=1).astype(np.int32).tolist()
            bot = np.mean(boxes[:, 2:4, :], axis=1).astype(np.int32).tolist()
            edge1 = select_edge(top + bot[::-1], boxes[0])
            edge2 = select_edge(top + bot[::-1], boxes[-1])
            if edge1 is not None:
                top.insert(0, edge1[0])
                bot.insert(0, edge1[1])
            if edge2 is not None:
                top.append(edge2[0])
                bot.append(edge2[1])
            boundary_point = np.array(top + bot[::-1])

        elif boxes.shape[0] == 1:
            top = boxes[0, 0:2, :].astype(np.int32).tolist()
            bot = boxes[0, 2:4:-1, :].astype(np.int32).tolist()
            boundary_point = np.array(top + bot)

        if boundary_point is None:
            continue

        boundary_point = [p for p in boundary_point.flatten().tolist()]
        bbox_contours.append(boundary_point)

    return bbox_contours


def merge_text_comps(edges, scores, text_comps, link_thr):
    """Merge text components into text instance."""
    clusters = graph_propagation(edges, scores, link_thr)
    pred_labels = clusters2labels(clusters, text_comps.shape[0])
    text_comps, final_pred = remove_single(text_comps, pred_labels)
    boundaries = comps2boundary(text_comps, final_pred)

    return boundaries

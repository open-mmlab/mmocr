# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

import mmocr.utils as utils


def sort_vertex(points_x, points_y):
    """Sort box vertices in clockwise order from left-top first.

    Args:
        points_x (list[float]): x of four vertices.
        points_y (list[float]): y of four vertices.
    Returns:
        sorted_points_x (list[float]): x of sorted four vertices.
        sorted_points_y (list[float]): y of sorted four vertices.
    """
    assert utils.is_type_list(points_x, (float, int))
    assert utils.is_type_list(points_y, (float, int))
    assert len(points_x) == 4
    assert len(points_y) == 4
    vertices = np.stack((points_x, points_y), axis=-1).astype(np.float32)
    vertices = _sort_vertex(vertices)
    sorted_points_x = list(vertices[:, 0])
    sorted_points_y = list(vertices[:, 1])
    return sorted_points_x, sorted_points_y


def _sort_vertex(vertices):
    assert vertices.ndim == 2
    assert vertices.shape[-1] == 2
    N = vertices.shape[0]
    if N == 0:
        return vertices

    center = np.mean(vertices, axis=0)
    directions = vertices - center
    angles = np.arctan2(directions[:, 1], directions[:, 0])
    sort_idx = np.argsort(angles)
    vertices = vertices[sort_idx]

    left_top = np.min(vertices, axis=0)
    dists = np.linalg.norm(left_top - vertices, axis=-1, ord=2)
    lefttop_idx = np.argmin(dists)
    indexes = (np.arange(N, dtype=np.int) + lefttop_idx) % N
    return vertices[indexes]


def sort_vertex8(points):
    """Sort vertex with 8 points [x1 y1 x2 y2 x3 y3 x4 y4]"""
    assert len(points) == 8
    vertices = _sort_vertex(np.array(points, dtype=np.float32).reshape(-1, 2))
    sorted_box = list(vertices.flatten())
    return sorted_box

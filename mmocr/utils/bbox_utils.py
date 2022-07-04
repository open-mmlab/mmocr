# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike

from mmocr.utils.check_argument import is_type_list
from mmocr.utils.point_utils import point_distance, points_center


def rescale_bbox(bbox: np.ndarray,
                 scale_factor: Tuple[int, int],
                 mode: str = 'mul') -> np.ndarray:
    """Rescale a bounding box according to scale_factor.

    The behavior is different depending on the mode. When mode is 'mul', the
    coordinates will be multiplied by scale_factor, which is usually used in
    preprocessing transforms such as :func:`Resize`.
    The coordinates will be divided by scale_factor if mode is 'div'. It can be
    used in postprocessors to recover the bbox in the original image size.

    Args:
        bbox (ndarray): A bounding box [x1, y1, x2, y2].
        scale_factor (tuple(int, int)): (w_scale, h_scale).
        model (str): Rescale mode. Can be 'mul' or 'div'. Defaults to 'mul'.

    Returns:
        np.ndarray: Rescaled bbox.
    """
    assert mode in ['mul', 'div']
    bbox = np.array(bbox, dtype=np.float32)
    bbox_shape = bbox.shape
    reshape_bbox = bbox.reshape(-1, 2)
    scale_factor = np.array(scale_factor, dtype=float)
    if mode == 'div':
        scale_factor = 1 / scale_factor
    bbox = (reshape_bbox * scale_factor[None]).reshape(bbox_shape)
    return bbox


def rescale_bboxes(bboxes: np.ndarray,
                   scale_factor: Tuple[int, int],
                   mode: str = 'mul') -> np.ndarray:
    """Rescale bboxes according to scale_factor.

    The behavior is different depending on the mode. When mode is 'mul', the
    coordinates will be multiplied by scale_factor, which is usually used in
    preprocessing transforms such as :func:`Resize`.
    The coordinates will be divided by scale_factor if mode is 'div'. It can be
    used in postprocessors to recover the bboxes in the original
    image size.

    Args:
        bboxes (np.ndarray]): Bounding bboxes in shape (N, 4)
        scale_factor (tuple(int, int)): (w_scale, h_scale).
        model (str): Rescale mode. Can be 'mul' or 'div'. Defaults to 'mul'.

    Returns:
        list[np.ndarray]: Rescaled bboxes.
    """
    bboxes = rescale_bbox(bboxes, scale_factor, mode)
    return bboxes


def bbox2poly(bbox: ArrayLike) -> np.array:
    """Converting a bounding box to a polygon.

    Args:
        bbox (ArrayLike): A bbox. In any form can be accessed by 1-D indices.
         E.g. list[float], np.ndarray, or torch.Tensor. bbox is written in
            [x1, y1, x2, y2].

    Returns:
        np.array: The converted polygon [x1, y1, x2, y1, x2, y2, x1, y2].
    """
    assert len(bbox) == 4
    return np.array([
        bbox[0], bbox[1], bbox[2], bbox[1], bbox[2], bbox[3], bbox[0], bbox[3]
    ])


def sort_vertex(points_x, points_y):
    # TODO Add typehints & docstring & test
    """Sort box vertices in clockwise order from left-top first.

    Args:
        points_x (list[float]): x of four vertices.
        points_y (list[float]): y of four vertices.
    Returns:
        sorted_points_x (list[float]): x of sorted four vertices.
        sorted_points_y (list[float]): y of sorted four vertices.
    """
    assert is_type_list(points_x, (float, int))
    assert is_type_list(points_y, (float, int))
    assert len(points_x) == 4
    assert len(points_y) == 4
    vertices = np.stack((points_x, points_y), axis=-1).astype(np.float32)
    vertices = _sort_vertex(vertices)
    sorted_points_x = list(vertices[:, 0])
    sorted_points_y = list(vertices[:, 1])
    return sorted_points_x, sorted_points_y


def _sort_vertex(vertices):
    # TODO Add typehints & docstring & test
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
    # TODO Add typehints & docstring & test
    """Sort vertex with 8 points [x1 y1 x2 y2 x3 y3 x4 y4]"""
    assert len(points) == 8
    vertices = _sort_vertex(np.array(points, dtype=np.float32).reshape(-1, 2))
    sorted_box = list(vertices.flatten())
    return sorted_box


def box_center_distance(b1, b2):
    # TODO typehints & docstring & test
    assert isinstance(b1, np.ndarray)
    assert isinstance(b2, np.ndarray)
    return point_distance(points_center(b1), points_center(b2))


def box_diag(box):
    # TODO typehints & docstring & test
    assert isinstance(box, np.ndarray)
    assert box.size == 8

    return point_distance(box[0:2], box[4:6])

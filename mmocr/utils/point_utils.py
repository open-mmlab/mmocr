# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmocr.utils.typing_utils import ArrayLike


def points_center(points: ArrayLike) -> np.ndarray:
    """Calculate the center of a set of points.

    Args:
        points (ArrayLike): A set of points.

    Returns:
        np.ndarray: The coordinate of center point.
    """
    points = np.array(points, dtype=np.float32)
    assert points.size % 2 == 0

    points = points.reshape([-1, 2])
    return np.mean(points, axis=0)


def point_distance(pt1: ArrayLike, pt2: ArrayLike) -> float:
    """Calculate the distance between two points.

    Args:
        pt1 (ArrayLike): The first point.
        pt2 (ArrayLike): The second point.

    Returns:
        float: The distance between two points.
    """
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)

    assert (pt1.size == 2 and pt2.size == 2)

    dist = np.square(pt2 - pt1).sum()
    dist = np.sqrt(dist)
    return dist

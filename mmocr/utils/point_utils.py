# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np


def dist_points2line(xs, ys, pt1, pt2):
    """Compute distances from points to a line. This is adapted from
    https://github.com/MhLiao/DB.

    Args:
        xs (ndarray): The x coordinates of points of size :math:`(N, )`.
        ys (ndarray): The y coordinates of size :math:`(N, )`.
        pt1 (ndarray): The first point on the line of size :math:`(2, )`.
        pt2 (ndarray): The second point on the line of size :math:`(2, )`.

    Returns:
        result (ndarray): The distance matrix of size :math:`(N, )`.
    """
    # suppose a triangle with three edge abc with c=point_1 point_2
    # a^2
    a_square = np.square(xs - pt1[0]) + np.square(ys - pt1[1])
    # b^2
    b_square = np.square(xs - pt2[0]) + np.square(ys - pt2[1])
    # c^2
    c_square = np.square(pt1[0] - pt2[0]) + np.square(pt1[1] - pt2[1])
    # -cosC=(c^2-a^2-b^2)/2(ab)
    neg_cos_c = ((c_square - a_square - b_square) /
                 (np.finfo(np.float32).eps + 2 * np.sqrt(a_square * b_square)))
    # sinC^2=1-cosC^2
    square_sin = 1 - np.square(neg_cos_c)
    square_sin = np.nan_to_num(square_sin)
    # distance=a*b*sinC/c=a*h/c=2*area/c
    result = np.sqrt(a_square * b_square * square_sin /
                     (np.finfo(np.float32).eps + c_square))
    # set result to minimum edge if C<pi/2
    result[neg_cos_c < 0] = np.sqrt(np.fmin(a_square, b_square))[neg_cos_c < 0]
    return result


def points_center(points):
    # TODO typehints & docstring
    assert isinstance(points, np.ndarray)
    assert points.size % 2 == 0

    points = points.reshape([-1, 2])
    return np.mean(points, axis=0)


def point_distance(p1, p2):
    # TODO typehints & docstring
    assert isinstance(p1, np.ndarray)
    assert isinstance(p2, np.ndarray)

    assert p1.size == 2
    assert p2.size == 2

    dist = np.square(p2 - p1)
    dist = np.sum(dist)
    dist = np.sqrt(dist)
    return dist

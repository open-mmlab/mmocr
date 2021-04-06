import numpy as np
from shapely.geometry import LineString, Point, Polygon

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
    assert utils.is_type_list(points_x, float) or utils.is_type_list(
        points_x, int)
    assert utils.is_type_list(points_y, float) or utils.is_type_list(
        points_y, int)
    assert len(points_x) == 4
    assert len(points_y) == 4

    x = np.array(points_x)
    y = np.array(points_y)
    center_x = np.sum(x) * 0.25
    center_y = np.sum(y) * 0.25

    x_arr = np.array(x - center_x)
    y_arr = np.array(y - center_y)

    angle = np.arctan2(y_arr, x_arr) * 180.0 / np.pi
    sort_idx = np.argsort(angle)

    sorted_points_x, sorted_points_y = [], []
    for i in range(4):
        sorted_points_x.append(points_x[sort_idx[i]])
        sorted_points_y.append(points_y[sort_idx[i]])

    return convert_canonical(sorted_points_x, sorted_points_y)


def convert_canonical(points_x, points_y):
    """Make left-top be first.

    Args:
        points_x (list[float]): x of four vertices.
        points_y (list[float]): y of four vertices.
    Returns:
        sorted_points_x (list[float]): x of sorted four vertices.
        sorted_points_y (list[float]): y of sorted four vertices.
    """
    assert utils.is_type_list(points_x, float) or utils.is_type_list(
        points_x, int)
    assert utils.is_type_list(points_y, float) or utils.is_type_list(
        points_y, int)
    assert len(points_x) == 4
    assert len(points_y) == 4

    points = [Point(points_x[i], points_y[i]) for i in range(4)]

    polygon = Polygon([(p.x, p.y) for p in points])
    min_x, min_y, _, _ = polygon.bounds
    points_to_lefttop = [
        LineString([points[i], Point(min_x, min_y)]) for i in range(4)
    ]
    distances = np.array([line.length for line in points_to_lefttop])
    sort_dist_idx = np.argsort(distances)
    lefttop_idx = sort_dist_idx[0]

    if lefttop_idx == 0:
        point_orders = [0, 1, 2, 3]
    elif lefttop_idx == 1:
        point_orders = [1, 2, 3, 0]
    elif lefttop_idx == 2:
        point_orders = [2, 3, 0, 1]
    else:
        point_orders = [3, 0, 1, 2]

    sorted_points_x = [points_x[i] for i in point_orders]
    sorted_points_y = [points_y[j] for j in point_orders]

    return sorted_points_x, sorted_points_y

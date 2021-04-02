import cv2
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


def box_jitter(points_x, points_y, jitter_ratio_x=0.5, jitter_ratio_y=0.1):
    """Jitter on the coordinates of bounding box.

    Args:
        points_x (list[float | int]): List of y for four vertices.
        points_y (list[float | int]): List of x for four vertices.
        jitter_ratio_x (float): Horizontal jitter ratio relative to the height.
        jitter_ratio_y (float): Vertical jitter ratio relative to the height.
    """
    assert len(points_x) == 4
    assert len(points_y) == 4
    assert isinstance(jitter_ratio_x, float)
    assert isinstance(jitter_ratio_y, float)
    assert 0 <= jitter_ratio_x < 1
    assert 0 <= jitter_ratio_y < 1

    points = [Point(points_x[i], points_y[i]) for i in range(4)]
    line_list = [
        LineString([points[i], points[i + 1 if i < 3 else 0]])
        for i in range(4)
    ]

    tmp_h = max(line_list[1].length, line_list[3].length)

    for i in range(4):
        jitter_pixel_x = (np.random.rand() - 0.5) * 2 * jitter_ratio_x * tmp_h
        jitter_pixel_y = (np.random.rand() - 0.5) * 2 * jitter_ratio_y * tmp_h
        points_x[i] += jitter_pixel_x
        points_y[i] += jitter_pixel_y


def warp_img(src_img,
             box,
             jitter_flag=False,
             jitter_ratio_x=0.5,
             jitter_ratio_y=0.1):
    """Crop box area from image using opencv warpPerspective w/o box jitter.

    Args:
        src_img (np.array): Image before cropping.
        box (list[float | int]): Coordinates of quadrangle.
    """
    assert utils.is_type_list(box, float) or utils.is_type_list(box, int)
    assert len(box) == 8

    h, w = src_img.shape[:2]
    points_x = [min(max(x, 0), w) for x in box[0:8:2]]
    points_y = [min(max(y, 0), h) for y in box[1:9:2]]

    points_x, points_y = sort_vertex(points_x, points_y)

    if jitter_flag:
        box_jitter(
            points_x,
            points_y,
            jitter_ratio_x=jitter_ratio_x,
            jitter_ratio_y=jitter_ratio_y)

    points = [Point(points_x[i], points_y[i]) for i in range(4)]
    edges = [
        LineString([points[i], points[i + 1 if i < 3 else 0]])
        for i in range(4)
    ]

    pts1 = np.float32([[points[i].x, points[i].y] for i in range(4)])
    box_width = max(edges[0].length, edges[2].length)
    box_height = max(edges[1].length, edges[3].length)

    pts2 = np.float32([[0, 0], [box_width, 0], [box_width, box_height],
                       [0, box_height]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst_img = cv2.warpPerspective(src_img, M,
                                  (int(box_width), int(box_height)))

    return dst_img


def crop_img(src_img, box):
    """Crop box area to rectangle.

    Args:
        src_img (np.array): Image before crop.
        box (list[float | int]): Points of quadrangle.
    """
    assert utils.is_type_list(box, float) or utils.is_type_list(box, int)
    assert len(box) == 8

    h, w = src_img.shape[:2]
    points_x = [min(max(x, 0), w) for x in box[0:8:2]]
    points_y = [min(max(y, 0), h) for y in box[1:9:2]]

    left = int(min(points_x))
    top = int(min(points_y))
    right = int(max(points_x))
    bottom = int(max(points_y))

    dst_img = src_img[top:bottom, left:right]

    return dst_img

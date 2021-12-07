# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
from shapely.geometry import LineString, Point

import mmocr.utils as utils
from .box_utils import sort_vertex


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
    assert utils.is_type_list(box, (float, int))
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


def crop_img(src_img, box, long_edge_pad_ratio=0.4, short_edge_pad_ratio=0.2):
    """Crop text region with their bounding box.

    Args:
        src_img (np.array): The original image.
        box (list[float | int]): Points of quadrangle.
        long_edge_pad_ratio (float): Box pad ratio for long edge
            corresponding to font size.
        short_edge_pad_ratio (float): Box pad ratio for short edge
            corresponding to font size.
    """
    assert utils.is_type_list(box, (float, int))
    assert len(box) == 8
    assert 0. <= long_edge_pad_ratio < 1.0
    assert 0. <= short_edge_pad_ratio < 1.0

    h, w = src_img.shape[:2]
    points_x = np.clip(np.array(box[0::2]), 0, w)
    points_y = np.clip(np.array(box[1::2]), 0, h)

    box_width = np.max(points_x) - np.min(points_x)
    box_height = np.max(points_y) - np.min(points_y)
    font_size = min(box_height, box_width)

    if box_height < box_width:
        horizontal_pad = long_edge_pad_ratio * font_size
        vertical_pad = short_edge_pad_ratio * font_size
    else:
        horizontal_pad = short_edge_pad_ratio * font_size
        vertical_pad = long_edge_pad_ratio * font_size

    left = np.clip(int(np.min(points_x) - horizontal_pad), 0, w)
    top = np.clip(int(np.min(points_y) - vertical_pad), 0, h)
    right = np.clip(int(np.max(points_x) + horizontal_pad), 0, w)
    bottom = np.clip(int(np.max(points_y) + vertical_pad), 0, h)

    dst_img = src_img[top:bottom, left:right]

    return dst_img

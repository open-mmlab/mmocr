# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike
from shapely.geometry import Polygon

from mmocr.utils import bbox2poly


def rescale_polygon(polygon: ArrayLike,
                    scale_factor: Tuple[int, int],
                    mode: str = 'mul') -> np.ndarray:
    """Rescale a polygon according to scale_factor.

    The behavior is different depending on the mode. When mode is 'mul', the
    coordinates will be multiplied by scale_factor, which is usually used in
    preprocessing transforms such as :func:`Resize`.
    The coordinates will be divided by scale_factor if mode is 'div'. It can be
    used in postprocessors to recover the polygon in the original
    image size.

    Args:
        polygon (ArrayLike): A polygon. In any form can be converted
            to an 1-D numpy array. E.g. list[float], np.ndarray,
            or torch.Tensor. Polygon is written in
            [x1, y1, x2, y2, ...].
        scale_factor (tuple(int, int)): (w_scale, h_scale).
        model (str): Rescale mode. Can be 'mul' or 'div'. Defaults to 'mul'.

    Returns:
        np.ndarray: Rescaled polygon.
    """
    assert len(polygon) % 2 == 0
    assert mode in ['mul', 'div']
    polygon = np.array(polygon, dtype=np.float32)
    poly_shape = polygon.shape
    reshape_polygon = polygon.reshape(-1, 2)
    scale_factor = np.array(scale_factor, dtype=float)
    if mode == 'div':
        scale_factor = 1 / scale_factor
    polygon = (reshape_polygon * scale_factor[None]).reshape(poly_shape)
    return polygon


def rescale_polygons(polygons: Sequence[ArrayLike],
                     scale_factor: Tuple[int, int],
                     mode: str = 'mul') -> Sequence[np.ndarray]:
    """Rescale polygons according to scale_factor.

    The behavior is different depending on the mode. When mode is 'mul', the
    coordinates will be multiplied by scale_factor, which is usually used in
    preprocessing transforms such as :func:`Resize`.
    The coordinates will be divided by scale_factor if mode is 'div'. It can be
    used in postprocessors to recover the polygon in the original
    image size.

    Args:
        polygons (list[ArrayLike]): A list of polygons, each written in
            [x1, y1, x2, y2, ...] and in any form can be converted
            to an 1-D numpy array. E.g. list[list[float]],
            list[np.ndarray], or list[torch.Tensor].
        scale_factor (tuple(int, int)): (w_scale, h_scale).
        model (str): Rescale mode. Can be 'mul' or 'div'. Defaults to 'mul'.

    Returns:
        list[np.ndarray]: Rescaled polygons.
    """
    results = []
    for polygon in polygons:
        results.append(rescale_polygon(polygon, scale_factor, mode))
    return results


def poly2bbox(polygon: ArrayLike) -> np.array:
    """Converting a polygon to a bounding box.

    Args:
         polygon (ArrayLike): A polygon. In any form can be converted
             to an 1-D numpy array. E.g. list[float], np.ndarray,
             or torch.Tensor. Polygon is written in
             [x1, y1, x2, y2, ...].

     Returns:
         np.array: The converted bounding box [x1, y1, x2, y2]
    """
    assert len(polygon) % 2 == 0
    polygon = np.array(polygon, dtype=np.float32)
    x = polygon[::2]
    y = polygon[1::2]
    return np.array([min(x), min(y), max(x), max(y)])


def crop_polygon(polygon: ArrayLike, crop_box: np.ndarray) -> np.ndarray:
    """Crop polygon to be within a box region.

    Args:
        polygon (ndarray): polygon in shape (N, ).
        crop_box (ndarray): target box region in shape (4, ).

    Returns:
        np.array or None: Cropped polygon.
    """
    polygon = np.asarray(polygon, dtype=np.float32)
    crop_box = np.asarray(crop_box, dtype=np.float32)
    poly = Polygon(polygon.reshape(-1, 2))
    crop_poly = Polygon(bbox2poly(crop_box).reshape(-1, 2))
    poly_cropped = poly.intersection(crop_poly)
    if poly_cropped.area == 0.:
        # If polygon is outside crop_box region, return None.
        return None
    else:
        poly_cropped = np.array(poly_cropped.boundary.xy)[:, :-1]
        return poly_cropped.reshape(-1)

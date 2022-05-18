# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike


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
        scale_factor (tuple(int, int)): (w_scale, h_scale)
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
        polygon (list[ArrayLike]): A list of polygons, each written in
            [x1, y1, x2, y2, ...] and in any form can be converted
            to an 1-D numpy array. E.g. list[list[float]],
            list[np.ndarray], or list[torch.Tensor].
        scale_factor (tuple(int, int)): (w_scale, h_scale)
        model (str): Rescale mode. Can be 'mul' or 'div'. Defaults to 'mul'.

    Returns:
        list[np.ndarray]: Rescaled polygons.
    """
    results = []
    for polygon in polygons:
        results.append(rescale_polygon(polygon, scale_factor, mode))
    return results

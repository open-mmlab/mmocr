# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike


def rescale_polygon(polygon: ArrayLike,
                    scale_factor: Tuple[int, int]) -> np.ndarray:
    """Rescale a polygon according to scale_factor.

    Args:
        polygon (ArrayLike): A polygon. In any form can be converted
            to an 1-D numpy array. E.g. list[float], np.ndarray,
            or torch.Tensor. Polygon is written in
            [x1, y1, x2, y2, ...].
        scale_factor (tuple(int, int)): (w_scale, h_scale)

    Returns:
        np.ndarray: Rescaled polygon.
    """
    assert len(polygon) % 2 == 0
    polygon = np.array(polygon, dtype=np.float32)
    poly_shape = polygon.shape
    reshape_polygon = polygon.reshape(-1, 2)
    scale_factor = np.array(scale_factor, dtype=float)
    polygon = (reshape_polygon / scale_factor[None]).reshape(poly_shape)
    return polygon


def rescale_polygons(polygons: Sequence[ArrayLike],
                     scale_factor: Tuple[int, int]) -> Sequence[np.ndarray]:
    """Rescale polygons according to scale_factor.

    Args:
        polygon (list[ArrayLike]): A list of polygons, each written in
            [x1, y1, x2, y2, ...] and in any form can be converted
            to an 1-D numpy array. E.g. list[list[float]],
            list[np.ndarray], or list[torch.Tensor].
        scale_factor (tuple(int, int)): (w_scale, h_scale)

    Returns:
        list[np.ndarray]: Rescaled polygons.
    """
    results = []
    for polygon in polygons:
        results.append(rescale_polygon(polygon, scale_factor))
    return results

# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike


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

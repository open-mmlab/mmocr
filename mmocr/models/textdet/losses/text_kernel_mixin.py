# Copyright (c) OpenMMLab. All rights reserved.
import sys
from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from mmengine.logging import MMLogger
from shapely.geometry import Polygon

from mmocr.utils.polygon_utils import offset_polygon


class TextKernelMixin:
    """Mixin class for text detection models that use text instance kernels."""

    def _generate_kernels(
        self,
        img_size: Tuple[int, int],
        text_polys: Sequence[np.ndarray],
        shrink_ratio: float,
        max_shrink_dist: Union[float, int] = sys.maxsize,
        ignore_flags: Optional[torch.Tensor] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate text instance kernels according to a shrink ratio.

        Args:
            img_size (tuple(int, int)): The image size of (height, width).
            text_polys (Sequence[np.ndarray]): 2D array of text polygons.
            shrink_ratio (float or int): The shrink ratio of kernel.
            max_shrink_dist (float or int): The maximum shrinking distance.
            ignore_flags (torch.BoolTensor, options): Indicate whether the
                corresponding text polygon is ignored.

        Returns:
            tuple(ndarray, ndarray): The text instance kernels of shape
                (height, width) and updated ignorance flags.
        """
        assert isinstance(img_size, tuple)
        assert isinstance(shrink_ratio, (float, int))

        logger: MMLogger = MMLogger.get_current_instance()

        h, w = img_size
        text_kernel = np.zeros((h, w), dtype=np.float32)

        for text_ind, poly in enumerate(text_polys):
            if ignore_flags is not None and ignore_flags[text_ind]:
                continue
            poly = poly.reshape(-1, 2).astype(np.int32)
            poly_obj = Polygon(poly)
            area = poly_obj.area
            peri = poly_obj.length
            distance = min(
                int(area * (1 - shrink_ratio * shrink_ratio) / (peri + 0.001) +
                    0.5), max_shrink_dist)
            shrunk_poly = offset_polygon(poly, -distance)

            if len(shrunk_poly) == 0:
                if ignore_flags is not None:
                    ignore_flags[text_ind] = True
                continue

            try:
                shrunk_poly = shrunk_poly.reshape(-1, 2)
            except Exception as e:
                logger.info(f'{shrunk_poly} with error {e}')
                if ignore_flags is not None:
                    ignore_flags[text_ind] = True
                continue

            cv2.fillPoly(text_kernel, [shrunk_poly.astype(np.int32)],
                         text_ind + 1)

        return text_kernel, ignore_flags

    def _generate_effective_mask(self, mask_size: Tuple[int, int],
                                 ignored_polygons: Sequence[np.ndarray]
                                 ) -> np.ndarray:
        """Generate effective mask by setting the invalid regions to 0 and 1
        otherwise.

        Args:
            mask_size (tuple(int, int)): The mask size.
            ignored_polygons (Sequence[ndarray]): 2-d array, representing all
                the ignored polygons of the text region.

        Returns:
            mask (ndarray): The effective mask of shape (height, width).
        """

        mask = np.ones(mask_size, dtype=np.uint8)

        for poly in ignored_polygons:
            instance = poly.reshape(-1, 2).astype(np.int32).reshape(1, -1, 2)
            cv2.fillPoly(mask, instance, 0)

        return mask

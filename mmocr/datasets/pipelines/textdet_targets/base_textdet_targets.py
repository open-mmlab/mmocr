import sys

import cv2
import numpy as np
import Polygon as plg
import pyclipper
from mmcv.utils import print_log

import mmocr.utils.check_argument as check_argument


class BaseTextDetTargets:
    """Generate text detector ground truths."""

    def __init__(self):
        pass

    def point2line(self, xs, ys, point_1, point_2):
        """Compute the distance from point to a line. This is adapted from
        https://github.com/MhLiao/DB.

        Args:
            xs (ndarray): The x coordinates of size hxw.
            ys (ndarray): The y coordinates of size hxw.
            point_1 (ndarray): The first point with shape 1x2.
            point_2 (ndarray): The second point with shape 1x2.

        Returns:
            result (ndarray): The distance matrix of size hxw.
        """
        # suppose a triangle with three edge abc with c=point_1 point_2
        # a^2
        square_distance_1 = np.square(xs - point_1[0]) + np.square(ys -
                                                                   point_1[1])
        # b^2
        square_distance_2 = np.square(xs - point_2[0]) + np.square(ys -
                                                                   point_2[1])
        # c^2
        square_distance = np.square(point_1[0] -
                                    point_2[0]) + np.square(point_1[1] -
                                                            point_2[1])
        # cosC=(c^2-a^2-b^2)/2(ab)
        cosin = (square_distance - square_distance_1 - square_distance_2) / \
            (np.finfo(np.float32).eps +
                2 * np.sqrt(square_distance_1 * square_distance_2))
        # sinC^2=1-cosC^2
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
        # distance=a*b*sinC/c=a*h/c=2*area/c
        result = np.sqrt(square_distance_1 * square_distance_2 * square_sin /
                         (np.finfo(np.float32).eps + square_distance))
        # set result to minimum edge if C>pi/2
        result[cosin < 0] = np.sqrt(
            np.fmin(square_distance_1, square_distance_2))[cosin < 0]
        return result

    def polygon_area(self, polygon):
        """Compute the polygon area. Please refer to Green's theorem.
        https://en.wikipedia.org/wiki/Green%27s_theorem. This is adapted from
        https://github.com/MhLiao/DB.

        Args:
            polygon (ndarray): The polygon boundary points.
        """

        polygon = polygon.reshape(-1, 2)
        edge = 0
        for i in range(polygon.shape[0]):
            next_index = (i + 1) % polygon.shape[0]
            edge += (polygon[next_index, 0] - polygon[i, 0]) * (
                polygon[next_index, 1] + polygon[i, 1])

        return edge / 2.

    def polygon_size(self, polygon):
        """Estimate the height and width of the minimum bounding box of the
        polygon.

        Args:
            polygon (ndarray): The polygon point sequence.

        Returns:
            size (tuple): The height and width of the minimum bounding box.
        """
        poly = polygon.reshape(-1, 2)
        rect = cv2.minAreaRect(poly.astype(np.int32))
        size = rect[1]
        return size

    def generate_kernels(self,
                         img_size,
                         text_polys,
                         shrink_ratio,
                         max_shrink=sys.maxsize,
                         ignore_tags=None):
        """Generate text instance kernels for one shrink ratio.

        Args:
            img_size (tuple(int, int)): The image size of (height, width).
            text_polys (list[list[ndarray]]: The list of text polygons.
            shrink_ratio (float): The shrink ratio of kernel.

        Returns:
            text_kernel (ndarray): The text kernel mask of (height, width).
        """
        assert isinstance(img_size, tuple)
        assert check_argument.is_2dlist(text_polys)
        assert isinstance(shrink_ratio, float)

        h, w = img_size
        text_kernel = np.zeros((h, w), dtype=np.float32)

        for text_ind, poly in enumerate(text_polys):
            instance = poly[0].reshape(-1, 2).astype(np.int32)
            area = plg.Polygon(instance).area()
            peri = cv2.arcLength(instance, True)
            distance = min(
                int(area * (1 - shrink_ratio * shrink_ratio) / (peri + 0.001) +
                    0.5), max_shrink)
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(instance, pyclipper.JT_ROUND,
                        pyclipper.ET_CLOSEDPOLYGON)
            shrinked = np.array(pco.Execute(-distance))

            # check shrinked == [] or empty ndarray
            if len(shrinked) == 0 or shrinked.size == 0:
                if ignore_tags is not None:
                    ignore_tags[text_ind] = True
                continue
            try:
                shrinked = np.array(shrinked[0]).reshape(-1, 2)

            except Exception as e:
                print_log(f'{shrinked} with error {e}')
                if ignore_tags is not None:
                    ignore_tags[text_ind] = True
                continue
            cv2.fillPoly(text_kernel, [shrinked.astype(np.int32)],
                         text_ind + 1)
        return text_kernel, ignore_tags

    def generate_effective_mask(self, mask_size: tuple, polygons_ignore):
        """Generate effective mask by setting the ineffective regions to 0 and
        effective regions to 1.

        Args:
            mask_size (tuple): The mask size.
            polygons_ignore (list[[ndarray]]: The list of ignored text
                polygons.

        Returns:
            mask (ndarray): The effective mask of (height, width).
        """

        assert check_argument.is_2dlist(polygons_ignore)

        mask = np.ones(mask_size, dtype=np.uint8)

        for poly in polygons_ignore:
            instance = poly[0].reshape(-1,
                                       2).astype(np.int32).reshape(1, -1, 2)
            cv2.fillPoly(mask, instance, 0)

        return mask

    def generate_targets(self, results):
        raise NotImplementedError

    def __call__(self, results):
        results = self.generate_targets(results)
        return results

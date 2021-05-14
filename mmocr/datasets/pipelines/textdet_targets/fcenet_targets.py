import cv2
import numpy as np
from numpy.linalg import norm

import mmocr.utils.check_argument as check_argument
from mmdet.datasets.builder import PIPELINES
from .textsnake_targets import TextSnakeTargets


@PIPELINES.register_module()
class FCENetTargets(TextSnakeTargets):
    """Generate the ground truth targets of FCENet: Fourier Contour Embedding
    for Arbitrary-Shaped Text Detection.

    [https://arxiv.org/abs/2104.10442]

    Args:
        fourier_degree (int): The maximum fourier transform degree k.
        resample_step (float): The step size for resampling the text center
            line (TCL). It's better not to exceed half of the minimum width.
        center_region_shrink_ratio (float): The shrink ratio of text center
            region.
        level_size_divisors (tuple(int)): The downsample ratio on each level.
        level_proportion_range (tuple(tuple(int))): The range of text sizes
            assigned to each level.
    """

    def __init__(self,
                 fourier_degree=5,
                 resample_step=4.0,
                 center_region_shrink_ratio=0.3,
                 level_size_divisors=(8, 16, 32),
                 level_proportion_range=((0, 0.4), (0.3, 0.7), (0.6, 1.0))):

        super().__init__()
        assert isinstance(level_size_divisors, tuple)
        assert isinstance(level_proportion_range, tuple)
        assert len(level_size_divisors) == len(level_proportion_range)
        self.fourier_degree = fourier_degree
        self.resample_step = resample_step
        self.center_region_shrink_ratio = center_region_shrink_ratio
        self.level_size_divisors = level_size_divisors
        self.level_proportion_range = level_proportion_range

    def generate_center_region_mask(self, img_size, text_polys):
        """Generate text center region mask.

        Args:
            img_size (tuple): The image size of (height, width).
            text_polys (list[list[ndarray]]): The list of text polygons.

        Returns:
            center_region_mask (ndarray): The text center region mask.
        """

        assert isinstance(img_size, tuple)
        assert check_argument.is_2dlist(text_polys)

        h, w = img_size

        center_region_mask = np.zeros((h, w), np.uint8)

        center_region_boxes = []
        for poly in text_polys:
            assert len(poly) == 1
            polygon_points = poly[0].reshape(-1, 2)
            _, _, top_line, bot_line = self.reorder_poly_edge(polygon_points)
            resampled_top_line, resampled_bot_line = self.resample_sidelines(
                top_line, bot_line, self.resample_step)
            resampled_bot_line = resampled_bot_line[::-1]
            center_line = (resampled_top_line + resampled_bot_line) / 2

            line_head_shrink_len = norm(resampled_top_line[0] -
                                        resampled_bot_line[0]) / 4.0
            line_tail_shrink_len = norm(resampled_top_line[-1] -
                                        resampled_bot_line[-1]) / 4.0
            head_shrink_num = int(line_head_shrink_len // self.resample_step)
            tail_shrink_num = int(line_tail_shrink_len // self.resample_step)
            if len(center_line) > head_shrink_num + tail_shrink_num + 2:
                center_line = center_line[head_shrink_num:len(center_line) -
                                          tail_shrink_num]
                resampled_top_line = resampled_top_line[
                    head_shrink_num:len(resampled_top_line) - tail_shrink_num]
                resampled_bot_line = resampled_bot_line[
                    head_shrink_num:len(resampled_bot_line) - tail_shrink_num]

            for i in range(0, len(center_line) - 1):
                tl = center_line[i] + (resampled_top_line[i] - center_line[i]
                                       ) * self.center_region_shrink_ratio
                tr = center_line[i + 1] + (
                    resampled_top_line[i + 1] -
                    center_line[i + 1]) * self.center_region_shrink_ratio
                br = center_line[i + 1] + (
                    resampled_bot_line[i + 1] -
                    center_line[i + 1]) * self.center_region_shrink_ratio
                bl = center_line[i] + (resampled_bot_line[i] - center_line[i]
                                       ) * self.center_region_shrink_ratio
                current_center_box = np.vstack([tl, tr, br,
                                                bl]).astype(np.int32)
                center_region_boxes.append(current_center_box)

        cv2.fillPoly(center_region_mask, center_region_boxes, 1)
        return center_region_mask

    def resample_polygon(self, polygon, n=400):
        """Resample one polygon with n points on its boundary.

        Args:
            polygon (list[float]): The polygon need to be resampled.
            n (int): The number of resampled points.
        Returns:
            resampled_polygon (list[float]): The resampled polygon.
        """
        length = []

        for i in range(len(polygon)):
            p1 = polygon[i]
            if i == len(polygon) - 1:
                p2 = polygon[0]
            else:
                p2 = polygon[i + 1]
            length.append(((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5)

        total_length = sum(length)
        n_on_each_line = (np.array(length) / (total_length + 1e-8)) * n
        n_on_each_line = n_on_each_line.astype(np.int32)
        new_polygon = []

        for i in range(len(polygon)):
            num = n_on_each_line[i]
            p1 = polygon[i]
            if i == len(polygon) - 1:
                p2 = polygon[0]
            else:
                p2 = polygon[i + 1]

            if num == 0:
                continue

            dxdy = (p2 - p1) / num
            for j in range(num):
                point = p1 + dxdy * j
                new_polygon.append(point)

        return np.array(new_polygon)

    def normalize_polygon(self, polygon):
        """Normalize one polygon so that its start point is at right most.

        Args:
            polygon (list[float]): The origin polygon.
        Returns:
            new_polygon (lost[float]): The polygon with start point at right.
        """
        temp_polygon = polygon - polygon.mean(axis=0)
        x = np.abs(temp_polygon[:, 0])
        y = temp_polygon[:, 1]
        index_x = np.argsort(x)
        index_y = np.argmin(y[index_x[:8]])
        index = index_x[index_y]
        new_polygon = np.concatenate([polygon[index:], polygon[:index]])
        return new_polygon

    def fourier_transform(self, polygon, fourier_degree):
        """Perform fourier transformation to generate fourier coefficients ck
        from polygon.

        Args:
            polygon (ndarray): An input polygon.
            fourier_degree (int): The maximum fourier degree K.
        Returns:
            c (ndarray(complex)): Fourier coefficients.
        """
        points = polygon[:, 0] + polygon[:, 1] * 1j
        n = len(points)
        t = np.multiply([i / n for i in range(n)], -2 * np.pi * 1j)

        e = complex(np.e)
        c = np.zeros((2 * fourier_degree + 1, ), dtype='complex')

        for i in range(-fourier_degree, fourier_degree + 1):
            c[i + fourier_degree] = np.sum(points * np.power(e, i * t)) / n

        return c

    def clockwise(self, c, fourier_degree):
        """Make sure the polygon reconstructed from fourier coefficients c in
        the clockwise direction.

        Args:
            polygon (list[float]): The origin polygon.
        Returns:
            new_polygon (lost[float]): The polygon with clockwise point order.
        """
        if np.abs(c[fourier_degree + 1]) > np.abs(c[fourier_degree - 1]):
            return c
        elif np.abs(c[fourier_degree + 1]) < np.abs(c[fourier_degree - 1]):
            return c[::-1]
        else:
            if np.abs(c[fourier_degree + 2]) > np.abs(c[fourier_degree - 2]):
                return c
            else:
                return c[::-1]

    def cal_fourier_signature(self, polygon, fourier_degree):
        """Calculate fourier signature from input polygon.

        Args:
              polygon (ndarray): The input polygon.
              fourier_degree (int): The maximum fourier degree K.
        Returns:
              fourier_signature (ndarray): An array shaped (2k+1, 2) containing
                  real part and image part of 2k+1 fourier coefficients.
        """
        resampled_polygon = self.resample_polygon(polygon)
        resampled_polygon = self.normalize_polygon(resampled_polygon)

        fourier_coeff = self.fourier_transform(resampled_polygon,
                                               fourier_degree)
        fourier_coeff = self.clockwise(fourier_coeff, fourier_degree)

        real_part = np.real(fourier_coeff).reshape((-1, 1))
        image_part = np.imag(fourier_coeff).reshape((-1, 1))
        fourier_signature = np.hstack([real_part, image_part])

        return fourier_signature

    def generate_fourier_maps(self, img_size, text_polys):
        """Generate Fourier coefficient maps.

        Args:
            img_size (tuple): The image size of (height, width).
            text_polys (list[list[ndarray]]): The list of text polygons.

        Returns:
            fourier_real_map (ndarray): The Fourier coefficient real part maps.
            fourier_image_map (ndarray): The Fourier coefficient image part
                maps.
        """

        assert isinstance(img_size, tuple)
        assert check_argument.is_2dlist(text_polys)

        h, w = img_size
        k = self.fourier_degree
        real_map = np.zeros((k * 2 + 1, h, w), dtype=np.float32)
        imag_map = np.zeros((k * 2 + 1, h, w), dtype=np.float32)

        for poly in text_polys:
            assert len(poly) == 1
            text_instance = [[poly[0][i], poly[0][i + 1]]
                             for i in range(0, len(poly[0]), 2)]
            mask = np.zeros((h, w), dtype=np.uint8)
            polygon = np.array(text_instance).reshape((1, -1, 2))
            cv2.fillPoly(mask, polygon.astype(np.int32), 1)
            fourier_coeff = self.cal_fourier_signature(polygon[0], k)
            for i in range(-k, k + 1):
                if i != 0:
                    real_map[i + k, :, :] = mask * fourier_coeff[i + k, 0] + (
                        1 - mask) * real_map[i + k, :, :]
                    imag_map[i + k, :, :] = mask * fourier_coeff[i + k, 1] + (
                        1 - mask) * imag_map[i + k, :, :]
                else:
                    yx = np.argwhere(mask > 0.5)
                    k_ind = np.ones((len(yx)), dtype=np.int64) * k
                    y, x = yx[:, 0], yx[:, 1]
                    real_map[k_ind, y, x] = fourier_coeff[k, 0] - x
                    imag_map[k_ind, y, x] = fourier_coeff[k, 1] - y

        return real_map, imag_map

    def generate_level_targets(self, img_size, text_polys, ignore_polys):
        """Generate ground truth target on each level.

        Args:
            img_size (list[int]): Shape of input image.
            text_polys (list[list[ndarray]]): A list of ground truth polygons.
            ignore_polys (list[list[ndarray]]): A list of ignored polygons.
        Returns:
            level_maps (list(ndarray)): A list of ground target on each level.
        """
        h, w = img_size
        lv_size_divs = self.level_size_divisors
        lv_proportion_range = self.level_proportion_range
        lv_text_polys = [[] for i in range(len(lv_size_divs))]
        lv_ignore_polys = [[] for i in range(len(lv_size_divs))]
        level_maps = []
        for poly in text_polys:
            assert len(poly) == 1
            text_instance = [[poly[0][i], poly[0][i + 1]]
                             for i in range(0, len(poly[0]), 2)]
            polygon = np.array(text_instance, dtype=np.int).reshape((1, -1, 2))
            _, _, box_w, box_h = cv2.boundingRect(polygon)
            proportion = max(box_h, box_w) / (h + 1e-8)

            for ind, proportion_range in enumerate(lv_proportion_range):
                if proportion_range[0] < proportion < proportion_range[1]:
                    lv_text_polys[ind].append([poly[0] / lv_size_divs[ind]])

        for ignore_poly in ignore_polys:
            assert len(ignore_poly) == 1
            text_instance = [[ignore_poly[0][i], ignore_poly[0][i + 1]]
                             for i in range(0, len(ignore_poly[0]), 2)]
            polygon = np.array(text_instance, dtype=np.int).reshape((1, -1, 2))
            _, _, box_w, box_h = cv2.boundingRect(polygon)
            proportion = max(box_h, box_w) / (h + 1e-8)

            for ind, proportion_range in enumerate(lv_proportion_range):
                if proportion_range[0] < proportion < proportion_range[1]:
                    lv_text_polys[ind].append(
                        [ignore_poly[0] / lv_size_divs[ind]])

        for ind, size_divisor in enumerate(lv_size_divs):
            current_level_maps = []
            level_img_size = (h // size_divisor, w // size_divisor)

            text_region = self.generate_text_region_mask(
                level_img_size, lv_text_polys[ind])
            text_region = np.expand_dims(text_region, axis=0)
            current_level_maps.append(text_region)

            center_region = self.generate_center_region_mask(
                level_img_size, lv_text_polys[ind])
            center_region = np.expand_dims(center_region, axis=0)
            current_level_maps.append(center_region)

            effective_mask = self.generate_effective_mask(
                level_img_size, lv_ignore_polys[ind])
            effective_mask = np.expand_dims(effective_mask, axis=0)
            current_level_maps.append(effective_mask)

            fourier_real_map, fourier_image_maps = self.generate_fourier_maps(
                level_img_size, lv_text_polys[ind])
            current_level_maps.append(fourier_real_map)
            current_level_maps.append(fourier_image_maps)

            level_maps.append(np.concatenate(current_level_maps))

        return level_maps

    def generate_targets(self, results):
        """Generate the ground truth targets for FCENet.

        Args:
            results (dict): The input result dictionary.

        Returns:
            results (dict): The output result dictionary.
        """

        assert isinstance(results, dict)

        polygon_masks = results['gt_masks'].masks
        polygon_masks_ignore = results['gt_masks_ignore'].masks

        h, w, _ = results['img_shape']

        level_maps = self.generate_level_targets((h, w), polygon_masks,
                                                 polygon_masks_ignore)

        results['mask_fields'].clear()  # rm gt_masks encoded by polygons
        mapping = {
            'p3_maps': level_maps[0],
            'p4_maps': level_maps[1],
            'p5_maps': level_maps[2]
        }
        for key, value in mapping.items():
            results[key] = value

        return results

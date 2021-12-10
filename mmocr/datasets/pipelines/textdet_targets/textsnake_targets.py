# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
from mmdet.core import BitmapMasks
from mmdet.datasets.builder import PIPELINES
from numpy.linalg import norm

import mmocr.utils.check_argument as check_argument
from . import BaseTextDetTargets


@PIPELINES.register_module()
class TextSnakeTargets(BaseTextDetTargets):
    """Generate the ground truth targets of TextSnake: TextSnake: A Flexible
    Representation for Detecting Text of Arbitrary Shapes.

    [https://arxiv.org/abs/1807.01544]. This was partially adapted from
    https://github.com/princewang1994/TextSnake.pytorch.

    Args:
        orientation_thr (float): The threshold for distinguishing between
            head edge and tail edge among the horizontal and vertical edges
            of a quadrangle.
    """

    def __init__(self,
                 orientation_thr=2.0,
                 resample_step=4.0,
                 center_region_shrink_ratio=0.3):

        super().__init__()
        self.orientation_thr = orientation_thr
        self.resample_step = resample_step
        self.center_region_shrink_ratio = center_region_shrink_ratio

    def vector_angle(self, vec1, vec2):
        if vec1.ndim > 1:
            unit_vec1 = vec1 / (norm(vec1, axis=-1) + 1e-8).reshape((-1, 1))
        else:
            unit_vec1 = vec1 / (norm(vec1, axis=-1) + 1e-8)
        if vec2.ndim > 1:
            unit_vec2 = vec2 / (norm(vec2, axis=-1) + 1e-8).reshape((-1, 1))
        else:
            unit_vec2 = vec2 / (norm(vec2, axis=-1) + 1e-8)
        return np.arccos(
            np.clip(np.sum(unit_vec1 * unit_vec2, axis=-1), -1.0, 1.0))

    def vector_slope(self, vec):
        assert len(vec) == 2
        return abs(vec[1] / (vec[0] + 1e-8))

    def vector_sin(self, vec):
        assert len(vec) == 2
        return vec[1] / (norm(vec) + 1e-8)

    def vector_cos(self, vec):
        assert len(vec) == 2
        return vec[0] / (norm(vec) + 1e-8)

    def find_head_tail(self, points, orientation_thr):
        """Find the head edge and tail edge of a text polygon.

        Args:
            points (ndarray): The points composing a text polygon.
            orientation_thr (float): The threshold for distinguishing between
                head edge and tail edge among the horizontal and vertical edges
                of a quadrangle.

        Returns:
            head_inds (list): The indexes of two points composing head edge.
            tail_inds (list): The indexes of two points composing tail edge.
        """

        assert points.ndim == 2
        assert points.shape[0] >= 4
        assert points.shape[1] == 2
        assert isinstance(orientation_thr, float)

        if len(points) > 4:
            pad_points = np.vstack([points, points[0]])
            edge_vec = pad_points[1:] - pad_points[:-1]

            theta_sum = []
            adjacent_vec_theta = []
            for i, edge_vec1 in enumerate(edge_vec):
                adjacent_ind = [x % len(edge_vec) for x in [i - 1, i + 1]]
                adjacent_edge_vec = edge_vec[adjacent_ind]
                temp_theta_sum = np.sum(
                    self.vector_angle(edge_vec1, adjacent_edge_vec))
                temp_adjacent_theta = self.vector_angle(
                    adjacent_edge_vec[0], adjacent_edge_vec[1])
                theta_sum.append(temp_theta_sum)
                adjacent_vec_theta.append(temp_adjacent_theta)
            theta_sum_score = np.array(theta_sum) / np.pi
            adjacent_theta_score = np.array(adjacent_vec_theta) / np.pi
            poly_center = np.mean(points, axis=0)
            edge_dist = np.maximum(
                norm(pad_points[1:] - poly_center, axis=-1),
                norm(pad_points[:-1] - poly_center, axis=-1))
            dist_score = edge_dist / (np.max(edge_dist) + 1e-12)
            position_score = np.zeros(len(edge_vec))
            score = 0.5 * theta_sum_score + 0.15 * adjacent_theta_score
            score += 0.35 * dist_score
            if len(points) % 2 == 0:
                position_score[(len(score) // 2 - 1)] += 1
                position_score[-1] += 1
            score += 0.1 * position_score
            pad_score = np.concatenate([score, score])
            score_matrix = np.zeros((len(score), len(score) - 3))
            x = np.arange(len(score) - 3) / float(len(score) - 4)
            gaussian = 1. / (np.sqrt(2. * np.pi) * 0.5) * np.exp(-np.power(
                (x - 0.5) / 0.5, 2.) / 2)
            gaussian = gaussian / np.max(gaussian)
            for i in range(len(score)):
                score_matrix[i, :] = score[i] + pad_score[
                    (i + 2):(i + len(score) - 1)] * gaussian * 0.3

            head_start, tail_increment = np.unravel_index(
                score_matrix.argmax(), score_matrix.shape)
            tail_start = (head_start + tail_increment + 2) % len(points)
            head_end = (head_start + 1) % len(points)
            tail_end = (tail_start + 1) % len(points)

            if head_end > tail_end:
                head_start, tail_start = tail_start, head_start
                head_end, tail_end = tail_end, head_end
            head_inds = [head_start, head_end]
            tail_inds = [tail_start, tail_end]
        else:
            if self.vector_slope(points[1] - points[0]) + self.vector_slope(
                    points[3] - points[2]) < self.vector_slope(
                        points[2] - points[1]) + self.vector_slope(points[0] -
                                                                   points[3]):
                horizontal_edge_inds = [[0, 1], [2, 3]]
                vertical_edge_inds = [[3, 0], [1, 2]]
            else:
                horizontal_edge_inds = [[3, 0], [1, 2]]
                vertical_edge_inds = [[0, 1], [2, 3]]

            vertical_len_sum = norm(points[vertical_edge_inds[0][0]] -
                                    points[vertical_edge_inds[0][1]]) + norm(
                                        points[vertical_edge_inds[1][0]] -
                                        points[vertical_edge_inds[1][1]])
            horizontal_len_sum = norm(
                points[horizontal_edge_inds[0][0]] -
                points[horizontal_edge_inds[0][1]]) + norm(
                    points[horizontal_edge_inds[1][0]] -
                    points[horizontal_edge_inds[1][1]])

            if vertical_len_sum > horizontal_len_sum * orientation_thr:
                head_inds = horizontal_edge_inds[0]
                tail_inds = horizontal_edge_inds[1]
            else:
                head_inds = vertical_edge_inds[0]
                tail_inds = vertical_edge_inds[1]

        return head_inds, tail_inds

    def reorder_poly_edge(self, points):
        """Get the respective points composing head edge, tail edge, top
        sideline and bottom sideline.

        Args:
            points (ndarray): The points composing a text polygon.

        Returns:
            head_edge (ndarray): The two points composing the head edge of text
                polygon.
            tail_edge (ndarray): The two points composing the tail edge of text
                polygon.
            top_sideline (ndarray): The points composing top curved sideline of
                text polygon.
            bot_sideline (ndarray): The points composing bottom curved sideline
                of text polygon.
        """

        assert points.ndim == 2
        assert points.shape[0] >= 4
        assert points.shape[1] == 2

        head_inds, tail_inds = self.find_head_tail(points,
                                                   self.orientation_thr)
        head_edge, tail_edge = points[head_inds], points[tail_inds]

        pad_points = np.vstack([points, points])
        if tail_inds[1] < 1:
            tail_inds[1] = len(points)
        sideline1 = pad_points[head_inds[1]:tail_inds[1]]
        sideline2 = pad_points[tail_inds[1]:(head_inds[1] + len(points))]
        sideline_mean_shift = np.mean(
            sideline1, axis=0) - np.mean(
                sideline2, axis=0)

        if sideline_mean_shift[1] > 0:
            top_sideline, bot_sideline = sideline2, sideline1
        else:
            top_sideline, bot_sideline = sideline1, sideline2

        return head_edge, tail_edge, top_sideline, bot_sideline

    def cal_curve_length(self, line):
        """Calculate the length of each edge on the discrete curve and the sum.

        Args:
            line (ndarray): The points composing a discrete curve.

        Returns:
            edges_length (ndarray): The length of each edge on the discrete
                curve.
            total_length (float): The total length of the discrete curve.
        """

        assert line.ndim == 2
        assert len(line) >= 2

        edges_length = np.sqrt((line[1:, 0] - line[:-1, 0])**2 +
                               (line[1:, 1] - line[:-1, 1])**2)
        total_length = np.sum(edges_length)
        return edges_length, total_length

    def resample_line(self, line, n):
        """Resample n points on a line.

        Args:
            line (ndarray): The points composing a line.
            n (int): The resampled points number.

        Returns:
            resampled_line (ndarray): The points composing the resampled line.
        """

        assert line.ndim == 2
        assert line.shape[0] >= 2
        assert line.shape[1] == 2
        assert isinstance(n, int)
        assert n > 2

        edges_length, total_length = self.cal_curve_length(line)
        t_org = np.insert(np.cumsum(edges_length), 0, 0)
        unit_t = total_length / (n - 1)
        t_equidistant = np.arange(1, n - 1, dtype=np.float32) * unit_t
        edge_ind = 0
        points = [line[0]]
        for t in t_equidistant:
            if edge_ind < len(edges_length) - 1:
                if t > t_org[edge_ind + 1]:
                    edge_ind += 1
            t_l, t_r = t_org[edge_ind], t_org[edge_ind + 1]
            weight = np.array([t_r - t, t - t_l], dtype=np.float32) / (
                t_r - t_l + 1e-8)
            p_coords = np.dot(weight, line[[edge_ind, edge_ind + 1]])
            points.append(p_coords)
        points.append(line[-1])
        resampled_line = np.vstack(points)

        return resampled_line

    def resample_sidelines(self, sideline1, sideline2, resample_step):
        """Resample two sidelines to be of the same points number according to
        step size.

        Args:
            sideline1 (ndarray): The points composing a sideline of a text
                polygon.
            sideline2 (ndarray): The points composing another sideline of a
                text polygon.
            resample_step (float): The resampled step size.

        Returns:
            resampled_line1 (ndarray): The resampled line 1.
            resampled_line2 (ndarray): The resampled line 2.
        """

        assert sideline1.ndim == sideline2.ndim == 2
        assert sideline1.shape[1] == sideline2.shape[1] == 2
        assert sideline1.shape[0] >= 2
        assert sideline2.shape[0] >= 2
        assert isinstance(resample_step, float)

        _, length1 = self.cal_curve_length(sideline1)
        _, length2 = self.cal_curve_length(sideline2)

        avg_length = (length1 + length2) / 2
        resample_point_num = max(int(float(avg_length) / resample_step) + 1, 3)

        resampled_line1 = self.resample_line(sideline1, resample_point_num)
        resampled_line2 = self.resample_line(sideline2, resample_point_num)

        return resampled_line1, resampled_line2

    def draw_center_region_maps(self, top_line, bot_line, center_line,
                                center_region_mask, radius_map, sin_map,
                                cos_map, region_shrink_ratio):
        """Draw attributes on text center region.

        Args:
            top_line (ndarray): The points composing top curved sideline of
                text polygon.
            bot_line (ndarray): The points composing bottom curved sideline
                of text polygon.
            center_line (ndarray): The points composing the center line of text
                instance.
            center_region_mask (ndarray): The text center region mask.
            radius_map (ndarray): The map where the distance from point to
                sidelines will be drawn on for each pixel in text center
                region.
            sin_map (ndarray): The map where vector_sin(theta) will be drawn
                on text center regions. Theta is the angle between tangent
                line and vector (1, 0).
            cos_map (ndarray): The map where vector_cos(theta) will be drawn on
                text center regions. Theta is the angle between tangent line
                and vector (1, 0).
            region_shrink_ratio (float): The shrink ratio of text center.
        """

        assert top_line.shape == bot_line.shape == center_line.shape
        assert (center_region_mask.shape == radius_map.shape == sin_map.shape
                == cos_map.shape)
        assert isinstance(region_shrink_ratio, float)
        for i in range(0, len(center_line) - 1):

            top_mid_point = (top_line[i] + top_line[i + 1]) / 2
            bot_mid_point = (bot_line[i] + bot_line[i + 1]) / 2
            radius = norm(top_mid_point - bot_mid_point) / 2

            text_direction = center_line[i + 1] - center_line[i]
            sin_theta = self.vector_sin(text_direction)
            cos_theta = self.vector_cos(text_direction)

            tl = center_line[i] + (top_line[i] -
                                   center_line[i]) * region_shrink_ratio
            tr = center_line[i + 1] + (
                top_line[i + 1] - center_line[i + 1]) * region_shrink_ratio
            br = center_line[i + 1] + (
                bot_line[i + 1] - center_line[i + 1]) * region_shrink_ratio
            bl = center_line[i] + (bot_line[i] -
                                   center_line[i]) * region_shrink_ratio
            current_center_box = np.vstack([tl, tr, br, bl]).astype(np.int32)

            cv2.fillPoly(center_region_mask, [current_center_box], color=1)
            cv2.fillPoly(sin_map, [current_center_box], color=sin_theta)
            cv2.fillPoly(cos_map, [current_center_box], color=cos_theta)
            cv2.fillPoly(radius_map, [current_center_box], color=radius)

    def generate_center_mask_attrib_maps(self, img_size, text_polys):
        """Generate text center region mask and geometric attribute maps.

        Args:
            img_size (tuple): The image size of (height, width).
            text_polys (list[list[ndarray]]): The list of text polygons.

        Returns:
            center_region_mask (ndarray): The text center region mask.
            radius_map (ndarray): The distance map from each pixel in text
                center region to top sideline.
            sin_map (ndarray): The sin(theta) map where theta is the angle
                between vector (top point - bottom point) and vector (1, 0).
            cos_map (ndarray): The cos(theta) map where theta is the angle
                between vector (top point - bottom point) and vector (1, 0).
        """

        assert isinstance(img_size, tuple)
        assert check_argument.is_2dlist(text_polys)

        h, w = img_size

        center_region_mask = np.zeros((h, w), np.uint8)
        radius_map = np.zeros((h, w), dtype=np.float32)
        sin_map = np.zeros((h, w), dtype=np.float32)
        cos_map = np.zeros((h, w), dtype=np.float32)

        for poly in text_polys:
            assert len(poly) == 1
            text_instance = [[poly[0][i], poly[0][i + 1]]
                             for i in range(0, len(poly[0]), 2)]
            polygon_points = np.array(text_instance).reshape(-1, 2)

            n = len(polygon_points)
            keep_inds = []
            for i in range(n):
                if norm(polygon_points[i] -
                        polygon_points[(i + 1) % n]) > 1e-5:
                    keep_inds.append(i)
            polygon_points = polygon_points[keep_inds]

            _, _, top_line, bot_line = self.reorder_poly_edge(polygon_points)
            resampled_top_line, resampled_bot_line = self.resample_sidelines(
                top_line, bot_line, self.resample_step)
            resampled_bot_line = resampled_bot_line[::-1]
            center_line = (resampled_top_line + resampled_bot_line) / 2

            if self.vector_slope(center_line[-1] - center_line[0]) > 0.9:
                if (center_line[-1] - center_line[0])[1] < 0:
                    center_line = center_line[::-1]
                    resampled_top_line = resampled_top_line[::-1]
                    resampled_bot_line = resampled_bot_line[::-1]
            else:
                if (center_line[-1] - center_line[0])[0] < 0:
                    center_line = center_line[::-1]
                    resampled_top_line = resampled_top_line[::-1]
                    resampled_bot_line = resampled_bot_line[::-1]

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

            self.draw_center_region_maps(resampled_top_line,
                                         resampled_bot_line, center_line,
                                         center_region_mask, radius_map,
                                         sin_map, cos_map,
                                         self.center_region_shrink_ratio)

        return center_region_mask, radius_map, sin_map, cos_map

    def generate_text_region_mask(self, img_size, text_polys):
        """Generate text center region mask and geometry attribute maps.

        Args:
            img_size (tuple): The image size (height, width).
            text_polys (list[list[ndarray]]): The list of text polygons.

        Returns:
            text_region_mask (ndarray): The text region mask.
        """

        assert isinstance(img_size, tuple)
        assert check_argument.is_2dlist(text_polys)

        h, w = img_size
        text_region_mask = np.zeros((h, w), dtype=np.uint8)

        for poly in text_polys:
            assert len(poly) == 1
            text_instance = [[poly[0][i], poly[0][i + 1]]
                             for i in range(0, len(poly[0]), 2)]
            polygon = np.array(
                text_instance, dtype=np.int32).reshape((1, -1, 2))
            cv2.fillPoly(text_region_mask, polygon, 1)

        return text_region_mask

    def generate_targets(self, results):
        """Generate the gt targets for TextSnake.

        Args:
            results (dict): The input result dictionary.

        Returns:
            results (dict): The output result dictionary.
        """

        assert isinstance(results, dict)

        polygon_masks = results['gt_masks'].masks
        polygon_masks_ignore = results['gt_masks_ignore'].masks

        h, w, _ = results['img_shape']

        gt_text_mask = self.generate_text_region_mask((h, w), polygon_masks)
        gt_mask = self.generate_effective_mask((h, w), polygon_masks_ignore)

        (gt_center_region_mask, gt_radius_map, gt_sin_map,
         gt_cos_map) = self.generate_center_mask_attrib_maps((h, w),
                                                             polygon_masks)

        results['mask_fields'].clear()  # rm gt_masks encoded by polygons
        mapping = {
            'gt_text_mask': gt_text_mask,
            'gt_center_region_mask': gt_center_region_mask,
            'gt_mask': gt_mask,
            'gt_radius_map': gt_radius_map,
            'gt_sin_map': gt_sin_map,
            'gt_cos_map': gt_cos_map
        }
        for key, value in mapping.items():
            value = value if isinstance(value, list) else [value]
            results[key] = BitmapMasks(value, h, w)
            results['mask_fields'].append(key)

        return results

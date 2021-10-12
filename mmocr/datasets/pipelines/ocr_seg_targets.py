# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
from mmdet.core import BitmapMasks
from mmdet.datasets.builder import PIPELINES

import mmocr.utils.check_argument as check_argument
from mmocr.models.builder import build_convertor


@PIPELINES.register_module()
class OCRSegTargets:
    """Generate gt shrunk kernels for segmentation based OCR framework.

    Args:
        label_convertor (dict): Dictionary to construct label_convertor
            to convert char to index.
        attn_shrink_ratio (float): The area shrunk ratio
            between attention kernels and gt text masks.
        seg_shrink_ratio (float): The area shrunk ratio
            between segmentation kernels and gt text masks.
        box_type (str): Character box type, should be either
            'char_rects' or 'char_quads', with 'char_rects'
            for rectangle with ``xyxy`` style and 'char_quads'
            for quadrangle with ``x1y1x2y2x3y3x4y4`` style.
    """

    def __init__(self,
                 label_convertor=None,
                 attn_shrink_ratio=0.5,
                 seg_shrink_ratio=0.25,
                 box_type='char_rects',
                 pad_val=255):

        assert isinstance(attn_shrink_ratio, float)
        assert isinstance(seg_shrink_ratio, float)
        assert 0. < attn_shrink_ratio < 1.0
        assert 0. < seg_shrink_ratio < 1.0
        assert label_convertor is not None
        assert box_type in ('char_rects', 'char_quads')

        self.attn_shrink_ratio = attn_shrink_ratio
        self.seg_shrink_ratio = seg_shrink_ratio
        self.label_convertor = build_convertor(label_convertor)
        self.box_type = box_type
        self.pad_val = pad_val

    def shrink_char_quad(self, char_quad, shrink_ratio):
        """Shrink char box in style of quadrangle.

        Args:
            char_quad (list[float]): Char box with format
                [x1, y1, x2, y2, x3, y3, x4, y4].
            shrink_ratio (float): The area shrunk ratio
                between gt kernels and gt text masks.
        """
        points = [[char_quad[0], char_quad[1]], [char_quad[2], char_quad[3]],
                  [char_quad[4], char_quad[5]], [char_quad[6], char_quad[7]]]
        shrink_points = []
        for p_idx, point in enumerate(points):
            p1 = points[(p_idx + 3) % 4]
            p2 = points[(p_idx + 1) % 4]

            dist1 = self.l2_dist_two_points(p1, point)
            dist2 = self.l2_dist_two_points(p2, point)
            min_dist = min(dist1, dist2)

            v1 = [p1[0] - point[0], p1[1] - point[1]]
            v2 = [p2[0] - point[0], p2[1] - point[1]]

            temp_dist1 = (shrink_ratio * min_dist /
                          dist1) if min_dist != 0 else 0.
            temp_dist2 = (shrink_ratio * min_dist /
                          dist2) if min_dist != 0 else 0.

            v1 = [temp * temp_dist1 for temp in v1]
            v2 = [temp * temp_dist2 for temp in v2]

            shrink_point = [
                round(point[0] + v1[0] + v2[0]),
                round(point[1] + v1[1] + v2[1])
            ]
            shrink_points.append(shrink_point)

        poly = np.array(shrink_points)

        return poly

    def shrink_char_rect(self, char_rect, shrink_ratio):
        """Shrink char box in style of rectangle.

        Args:
            char_rect (list[float]): Char box with format
                [x_min, y_min, x_max, y_max].
            shrink_ratio (float): The area shrunk ratio
                between gt kernels and gt text masks.
        """
        x_min, y_min, x_max, y_max = char_rect
        w = x_max - x_min
        h = y_max - y_min
        x_min_s = round((x_min + x_max - w * shrink_ratio) / 2)
        y_min_s = round((y_min + y_max - h * shrink_ratio) / 2)
        x_max_s = round((x_min + x_max + w * shrink_ratio) / 2)
        y_max_s = round((y_min + y_max + h * shrink_ratio) / 2)
        poly = np.array([[x_min_s, y_min_s], [x_max_s, y_min_s],
                         [x_max_s, y_max_s], [x_min_s, y_max_s]])

        return poly

    def generate_kernels(self,
                         resize_shape,
                         pad_shape,
                         char_boxes,
                         char_inds,
                         shrink_ratio=0.5,
                         binary=True):
        """Generate char instance kernels for one shrink ratio.

        Args:
            resize_shape (tuple(int, int)): Image size (height, width)
                after resizing.
            pad_shape (tuple(int, int)):  Image size (height, width)
                after padding.
            char_boxes (list[list[float]]): The list of char polygons.
            char_inds (list[int]): List of char indexes.
            shrink_ratio (float): The shrink ratio of kernel.
            binary (bool): If True, return binary ndarray
                containing 0 & 1 only.
        Returns:
            char_kernel (ndarray): The text kernel mask of (height, width).
        """
        assert isinstance(resize_shape, tuple)
        assert isinstance(pad_shape, tuple)
        assert check_argument.is_2dlist(char_boxes)
        assert check_argument.is_type_list(char_inds, int)
        assert isinstance(shrink_ratio, float)
        assert isinstance(binary, bool)

        char_kernel = np.zeros(pad_shape, dtype=np.int32)
        char_kernel[:resize_shape[0], resize_shape[1]:] = self.pad_val

        for i, char_box in enumerate(char_boxes):
            if self.box_type == 'char_rects':
                poly = self.shrink_char_rect(char_box, shrink_ratio)
            elif self.box_type == 'char_quads':
                poly = self.shrink_char_quad(char_box, shrink_ratio)

            fill_value = 1 if binary else char_inds[i]
            cv2.fillConvexPoly(char_kernel, poly.astype(np.int32),
                               (fill_value))

        return char_kernel

    def l2_dist_two_points(self, p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    def __call__(self, results):
        img_shape = results['img_shape']
        resize_shape = results['resize_shape']

        h_scale = 1.0 * resize_shape[0] / img_shape[0]
        w_scale = 1.0 * resize_shape[1] / img_shape[1]

        char_boxes, char_inds = [], []
        char_num = len(results['ann_info'][self.box_type])
        for i in range(char_num):
            char_box = results['ann_info'][self.box_type][i]
            num_points = 2 if self.box_type == 'char_rects' else 4
            for j in range(num_points):
                char_box[j * 2] = round(char_box[j * 2] * w_scale)
                char_box[j * 2 + 1] = round(char_box[j * 2 + 1] * h_scale)
            char_boxes.append(char_box)
            char = results['ann_info']['chars'][i]
            char_ind = self.label_convertor.str2idx([char])[0][0]
            char_inds.append(char_ind)

        resize_shape = tuple(results['resize_shape'][:2])
        pad_shape = tuple(results['pad_shape'][:2])
        binary_target = self.generate_kernels(
            resize_shape,
            pad_shape,
            char_boxes,
            char_inds,
            shrink_ratio=self.attn_shrink_ratio,
            binary=True)

        seg_target = self.generate_kernels(
            resize_shape,
            pad_shape,
            char_boxes,
            char_inds,
            shrink_ratio=self.seg_shrink_ratio,
            binary=False)

        mask = np.ones(pad_shape, dtype=np.int32)
        mask[:resize_shape[0], resize_shape[1]:] = 0

        results['gt_kernels'] = BitmapMasks([binary_target, seg_target, mask],
                                            pad_shape[0], pad_shape[1])
        results['mask_fields'] = ['gt_kernels']

        return results

# Copyright (c) OpenMMLab. All rights reserved.
import math
import sys

import cv2
import numpy as np
import pyclipper
from mmcv.utils import print_log
from shapely.geometry import Polygon as plg

from mmocr.datasets.pipelines.textdet_targets.base_textdet_targets import \
    BaseTextDetTargets


class EASTTargets(BaseTextDetTargets):

    def __init__(self, box_type, scale, shrink_ratio=0.4):
        assert box_type in ['RBOX', 'QUAD']
        super().__init__()
        self.box_type = box_type
        self.scale = scale
        self.shrink_ratio = shrink_ratio

    def generate_targets(self, results):
        """Generate the ground truth targets for EAST.

        Args:
            results (dict): The input result dictionary.

        Returns:
            results (dict): The output result dictionary.
        """
        assert isinstance(results, dict)
        polys = results['gt_masks']
        polys_ignore = results['gt_masks_ignore']
        h, w, _ = results['img_shape']
        # generate shrink kernels
        # shrunk_polys :(n, 4, 2)
        shrunk_polys, _ = self.shrink_poly((h, w), polys, self.shrink_ratio)
        # gt_scores: mask of (h*scale, w*scale), 1 for text, 0 for background
        gt_scores = self.generate_score((h, w), shrunk_polys)
        # gt_scores_ignore: mask of (h*scale, w*scale), 1 for ignored text
        polys_ignore = np.array(polys_ignore).reshape(-1, 4, 2)
        gt_scores_ignore = self.generate_score((h, w), polys_ignore)
        # generate gt_bboxes
        if self.box_type == 'RBOX':
            gt_bboxes = self.generate_rbox((h, w), polys, shrunk_polys)
        elif self.box_type == 'QUAD':
            gt_bboxes = self.generate_quad((h, w), polys, shrunk_polys)
        results = {
            'gt_scores': gt_scores,
            'gt_scores_ignore': gt_scores_ignore,
            'gt_bboxes': gt_bboxes
        }
        return results

    def generate_score(self, img_shape, shrunk_polys):
        # shrink_polys = shrink_polys.reshape(-1, 4, 2)
        gt_scores = np.zeros((int(
            img_shape[0] * self.scale), int(img_shape[0] * self.scale), 1),
                             np.float32)
        gt_scores = cv2.fillPoly(
            gt_scores,
            np.around(shrunk_polys * self.scale).astype(np.int32), 1)
        return gt_scores

    def generate_rbox(self, img_shape, polys, shrunk_polys):
        gt_bboxes = np.zeros((5, int(
            img_shape[0] * self.scale), int(img_shape[1] * self.scale)),
                             dtype=np.float32)
        # convert poly to rotate bbox to caluate angle, w, h
        rboxes = []
        for poly in polys:
            rboxes.append(self.poly2rbox(poly))
        # convert rotate bbox back to polys to calculate vertex coordinates
        coords = []
        for rbox in rboxes:
            coords.append(self.rbox2poly(rbox))
        for coord, shrunk_poly, rbox in zip(coords, shrunk_polys, rboxes):
            temp_mask = np.zeros((int(
                img_shape[0] * self.scale), int(img_shape[1] * self.scale)),
                                 np.float32)
            # score mask at feature level
            shrunk_poly_mask = cv2.fillPoly(
                temp_mask,
                [np.around(shrunk_poly * self.scale).astype(np.int32)], 1)
            # feature level coordinates
            y_feat, x_feat = np.where(shrunk_poly_mask > 0)
            # image level coordinates
            x_img = np.around(x_feat / self.scale)
            y_img = np.around(y_feat / self.scale)
            top = self.point2line(x_img, y_img, coord[0], coord[1])
            right = self.point2line(x_img, y_img, coord[1], coord[2])
            down = self.point2line(x_img, y_img, coord[2], coord[3])
            left = self.point2line(x_img, y_img, coord[3], coord[0])

            x_coordinates = x_feat.astype(np.int32).tolist()
            y_coordinates = y_feat.astype(np.int32).tolist()
            gt_bboxes[0, x_coordinates, y_coordinates] = top
            gt_bboxes[1, x_coordinates, y_coordinates] = right
            gt_bboxes[2, x_coordinates, y_coordinates] = down
            gt_bboxes[3, x_coordinates, y_coordinates] = left
            gt_bboxes[4, x_coordinates, y_coordinates] = np.ones_like(
                top, dtype=np.float32) * rbox[-1]
        return gt_bboxes

    def generate_quad(self, vertices):
        pass

    def poly2rbox(self, poly):
        poly = np.array(poly).reshape(4, 2)
        rbbox = cv2.minAreaRect(poly)
        x = rbbox[0][0]
        y = rbbox[0][1]
        w = rbbox[1][0]
        h = rbbox[1][1]
        angle = rbbox[2]
        # case 1
        if w >= h:
            return ([x, y, w, h, angle])
        # case 2
        else:
            return ([x, y, h, w, angle - 90.])

    def rbox2poly(self, rbox):
        x, y, w, h, angle = rbox
        angle = angle / 180 * math.pi
        center_coords = np.array([
            -0.5 * w, -0.5 * h, 0.5 * w, -0.5 * h, 0.5 * w, 0.5 * h, -0.5 * w,
            0.5 * h
        ]).reshape(4, 2).T
        rotate_matrix = np.array([[math.cos(angle), -math.sin(angle)],
                                  [math.sin(angle),
                                   math.cos(angle)]])
        center_rotate_coords = np.dot(rotate_matrix, center_coords)
        rotate_coords = center_rotate_coords + np.array([[x], [y]])
        return rotate_coords.T.reshape(4, 2)

    def shrink_poly(self,
                    img_size,
                    text_polys,
                    shrink_ratio,
                    max_shrink=sys.maxsize,
                    ignore_tags=None):
        """Generate shrunk polys for one shrink ratio.

        Args:
            img_size (tuple(int, int)): The image size of (height, width).
            text_polys (list[list[ndarray]]: The list of text polygons.
            shrink_ratio (float): The shrink ratio of kernel.

        Returns:p
            shrunk (ndarray): shrunk polys
        """
        assert isinstance(img_size, tuple)
        assert isinstance(shrink_ratio, float)

        shrunks = []
        for text_ind, poly in enumerate(text_polys):
            instance = poly.reshape(-1, 2).astype(np.float32)
            area = plg(instance).area
            peri = cv2.arcLength(instance, True)
            distance = min(
                int(area * (1 - shrink_ratio * shrink_ratio) / (peri + 0.001) +
                    0.5), max_shrink)
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(instance, pyclipper.JT_ROUND,
                        pyclipper.ET_CLOSEDPOLYGON)
            shrunk = np.array(pco.Execute(-distance))

            # check shrunk == [] or empty ndarray
            if len(shrunk) == 0 or shrunk.size == 0:
                if ignore_tags is not None:
                    ignore_tags[text_ind] = True
                continue
            try:
                shrunk = np.array(shrunk[0]).reshape(-1, 2)
                shrunks.append(shrunk)
            except Exception as e:
                print_log(f'{shrunk} with error {e}')
                if ignore_tags is not None:
                    ignore_tags[text_ind] = True
                continue
        return np.array(shrunks), ignore_tags

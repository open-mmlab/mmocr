# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
from mmdet.core.mask import PolygonMasks

from mmocr.registry import TRANSFORMS


@TRANSFORMS.register_module()
class EastRandomCrop:

    def __init__(self,
                 target_size=(640, 640),
                 max_tries=10,
                 min_crop_side_ratio=0.1):
        self.target_size = target_size
        self.max_tries = max_tries
        self.min_crop_side_ratio = min_crop_side_ratio

    def __call__(self, results):
        # sampling crop
        # crop image, boxes, masks
        img = results['img']
        crop_x, crop_y, crop_w, crop_h = self.crop_area(
            img, results['gt_masks'])
        scale_w = self.target_size[0] / crop_w
        scale_h = self.target_size[1] / crop_h
        scale = min(scale_w, scale_h)
        h = int(crop_h * scale)
        w = int(crop_w * scale)
        padded_img = np.zeros(
            (self.target_size[1], self.target_size[0], img.shape[2]),
            img.dtype)
        padded_img[:h, :w] = mmcv.imresize(
            img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], (w, h))

        # for bboxes
        for key in results['bbox_fields']:
            lines = []
            for box in results[key]:
                box = box.reshape(2, 2)
                poly = ((box - (crop_x, crop_y)) * scale)
                if not self.is_poly_outside_rect(poly, 0, 0, w, h):
                    lines.append(poly.flatten())
            results[key] = np.array(lines)
        # for masks
        for key in results['mask_fields']:
            polys = []
            polys_label = []
            for poly in results[key]:
                poly = np.array(poly).reshape(-1, 2)
                poly = ((poly - (crop_x, crop_y)) * scale)
                if not self.is_poly_outside_rect(poly, 0, 0, w, h):
                    polys.append([poly])
                    polys_label.append(0)
            results[key] = PolygonMasks(polys, *self.target_size)
            if key == 'gt_masks':
                results['gt_labels'] = polys_label

        results['img'] = padded_img
        results['img_shape'] = padded_img.shape

        return results

    def is_poly_in_rect(self, poly, x, y, w, h):
        poly = np.array(poly)
        if poly[:, 0].min() < x or poly[:, 0].max() > x + w:
            return False
        if poly[:, 1].min() < y or poly[:, 1].max() > y + h:
            return False
        return True

    def is_poly_outside_rect(self, poly, x, y, w, h):
        poly = np.array(poly).reshape(-1, 2)
        if poly[:, 0].max() < x or poly[:, 0].min() > x + w:
            return True
        if poly[:, 1].max() < y or poly[:, 1].min() > y + h:
            return True
        return False

    def split_regions(self, axis):
        regions = []
        min_axis = 0
        for i in range(1, axis.shape[0]):
            if axis[i] != axis[i - 1] + 1:
                region = axis[min_axis:i]
                min_axis = i
                regions.append(region)
        return regions

    def random_select(self, axis, max_size):
        xx = np.random.choice(axis, size=2)
        xmin = np.min(xx)
        xmax = np.max(xx)
        xmin = np.clip(xmin, 0, max_size - 1)
        xmax = np.clip(xmax, 0, max_size - 1)
        return xmin, xmax

    def region_wise_random_select(self, regions):
        selected_index = list(np.random.choice(len(regions), 2))
        selected_values = []
        for index in selected_index:
            axis = regions[index]
            xx = int(np.random.choice(axis, size=1))
            selected_values.append(xx)
        xmin = min(selected_values)
        xmax = max(selected_values)
        return xmin, xmax

    def crop_area(self, img, polys):
        h, w, _ = img.shape
        h_array = np.zeros(h, dtype=np.int32)
        w_array = np.zeros(w, dtype=np.int32)
        for points in polys:
            points = np.round(
                points, decimals=0).astype(np.int32).reshape(-1, 2)
            min_x = np.min(points[:, 0])
            max_x = np.max(points[:, 0])
            w_array[min_x:max_x] = 1
            min_y = np.min(points[:, 1])
            max_y = np.max(points[:, 1])
            h_array[min_y:max_y] = 1
        # ensure the cropped area not across a text
        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]

        if len(h_axis) == 0 or len(w_axis) == 0:
            return 0, 0, w, h

        h_regions = self.split_regions(h_axis)
        w_regions = self.split_regions(w_axis)

        for i in range(self.max_tries):
            if len(w_regions) > 1:
                xmin, xmax = self.region_wise_random_select(w_regions)
            else:
                xmin, xmax = self.random_select(w_axis, w)
            if len(h_regions) > 1:
                ymin, ymax = self.region_wise_random_select(h_regions)
            else:
                ymin, ymax = self.random_select(h_axis, h)

            if (xmax - xmin < self.min_crop_side_ratio * w
                    or ymax - ymin < self.min_crop_side_ratio * h):
                # area too small
                continue
            num_poly_in_rect = 0
            for poly in polys:
                if not self.is_poly_outside_rect(poly, xmin, ymin, xmax - xmin,
                                                 ymax - ymin):
                    num_poly_in_rect += 1
                    break

            if num_poly_in_rect > 0:
                return xmin, ymin, xmax - xmin, ymax - ymin

        return 0, 0, w, h

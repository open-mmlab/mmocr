import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon

from mmdet.core import BitmapMasks
from mmdet.datasets.builder import PIPELINES
from . import BaseTextDetTargets


@PIPELINES.register_module()
class DBNetTargets(BaseTextDetTargets):
    """Generate gt shrinked text, gt threshold map, and their effective region
    masks to learn DBNet: Real-time Scene Text Detection with Differentiable
    Binarization [https://arxiv.org/abs/1911.08947]. This was partially adapted
    from https://github.com/MhLiao/DB.

    Args:
        shrink_ratio (float): The area shrinked ratio between text
            kernels and their text masks.
        thr_min (float): The minimum value of the threshold map.
        thr_max (float): The maximum value of the threshold map.
        min_short_size (int): The minimum size of polygon below which
            the polygon is invalid.
    """

    def __init__(self,
                 shrink_ratio=0.4,
                 thr_min=0.3,
                 thr_max=0.7,
                 min_short_size=8):
        super().__init__()
        self.shrink_ratio = shrink_ratio
        self.thr_min = thr_min
        self.thr_max = thr_max
        self.min_short_size = min_short_size

    def find_invalid(self, results):
        """Find invalid polygons.

        Args:
            results (dict): The dict containing gt_mask.

        Returns:
            ignore_tags (list[bool]): The indicators for ignoring polygons.
        """
        texts = results['gt_masks'].masks
        ignore_tags = [False] * len(texts)

        for inx, text in enumerate(texts):
            if self.invalid_polygon(text[0]):
                ignore_tags[inx] = True
        return ignore_tags

    def invalid_polygon(self, poly):
        """Judge the input polygon is invalid or not. It is invalid if its area
        smaller than 1 or the shorter side of its minimum bounding box smaller
        than min_short_size.

        Args:
            poly (ndarray): The polygon boundary point sequence.

        Returns:
            True/False (bool): Whether the polygon is invalid.
        """
        area = self.polygon_area(poly)
        if abs(area) < 1:
            return True
        short_size = min(self.polygon_size(poly))
        if short_size < self.min_short_size:
            return True

        return False

    def ignore_texts(self, results, ignore_tags):
        """Ignore gt masks and gt_labels while padding gt_masks_ignore in
        results given ignore_tags.

        Args:
            results (dict): Result for one image.
            ignore_tags (list[int]): Indicate whether to ignore its
                corresponding ground truth text.

        Returns:
            results (dict): Results after filtering.
        """
        flag_len = len(ignore_tags)
        assert flag_len == len(results['gt_masks'].masks)
        assert flag_len == len(results['gt_labels'])

        results['gt_masks_ignore'].masks += [
            mask for i, mask in enumerate(results['gt_masks'].masks)
            if ignore_tags[i]
        ]
        results['gt_masks'].masks = [
            mask for i, mask in enumerate(results['gt_masks'].masks)
            if not ignore_tags[i]
        ]
        results['gt_labels'] = np.array([
            mask for i, mask in enumerate(results['gt_labels'])
            if not ignore_tags[i]
        ])

        return results

    def generate_thr_map(self, img_size, polygons):
        """Generate threshold map.

        Args:
            img_size (tuple(int)): The image size (h,w)
            polygons (list(ndarray)): The polygon list.

        Returns:
            thr_map (ndarray): The generated threshold map.
            thr_mask (ndarray): The effective mask of threshold map.
        """
        thr_map = np.zeros(img_size, dtype=np.float32)
        thr_mask = np.zeros(img_size, dtype=np.uint8)

        for polygon in polygons:
            self.draw_border_map(polygon[0], thr_map, mask=thr_mask)
        thr_map = thr_map * (self.thr_max - self.thr_min) + self.thr_min

        return thr_map, thr_mask

    def draw_border_map(self, polygon, canvas, mask):
        """Generate threshold map for one polygon.

        Args:
            polygon(ndarray): The polygon boundary ndarray.
            canvas(ndarray): The generated threshold map.
            mask(ndarray): The generated threshold mask.
        """
        polygon = polygon.reshape(-1, 2)
        assert polygon.ndim == 2
        assert polygon.shape[1] == 2

        polygon_shape = Polygon(polygon)
        distance = polygon_shape.area * \
            (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(p) for p in polygon]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND,
                        pyclipper.ET_CLOSEDPOLYGON)
        padded_polygon = padding.Execute(distance)
        if len(padded_polygon) > 0:
            padded_polygon = np.array(padded_polygon[0])
        else:
            print(f'padding {polygon} with {distance} gets {padded_polygon}')
            padded_polygon = polygon.copy().astype(np.int32)
        cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

        x_min = padded_polygon[:, 0].min()
        x_max = padded_polygon[:, 0].max()
        y_min = padded_polygon[:, 1].min()
        y_max = padded_polygon[:, 1].max()
        width = x_max - x_min + 1
        height = y_max - y_min + 1

        polygon[:, 0] = polygon[:, 0] - x_min
        polygon[:, 1] = polygon[:, 1] - y_min

        xs = np.broadcast_to(
            np.linspace(0, width - 1, num=width).reshape(1, width),
            (height, width))
        ys = np.broadcast_to(
            np.linspace(0, height - 1, num=height).reshape(height, 1),
            (height, width))

        distance_map = np.zeros((polygon.shape[0], height, width),
                                dtype=np.float32)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            absolute_distance = self.point2line(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = distance_map.min(axis=0)

        x_min_valid = min(max(0, x_min), canvas.shape[1] - 1)
        x_max_valid = min(max(0, x_max), canvas.shape[1] - 1)
        y_min_valid = min(max(0, y_min), canvas.shape[0] - 1)
        y_max_valid = min(max(0, y_max), canvas.shape[0] - 1)
        canvas[y_min_valid:y_max_valid + 1,
               x_min_valid:x_max_valid + 1] = np.fmax(
                   1 - distance_map[y_min_valid - y_min:y_max_valid - y_max +
                                    height, x_min_valid - x_min:x_max_valid -
                                    x_max + width],
                   canvas[y_min_valid:y_max_valid + 1,
                          x_min_valid:x_max_valid + 1])

    def generate_targets(self, results):
        """Generate the gt targets for DBNet.

        Args:
            results (dict): The input result dictionary.

        Returns:
            results (dict): The output result dictionary.
        """
        assert isinstance(results, dict)
        polygons = results['gt_masks'].masks
        if 'bbox_fields' in results:
            results['bbox_fields'].clear()
        ignore_tags = self.find_invalid(results)
        h, w, _ = results['img_shape']

        gt_shrink, ignore_tags = self.generate_kernels((h, w),
                                                       polygons,
                                                       self.shrink_ratio,
                                                       ignore_tags=ignore_tags)

        results = self.ignore_texts(results, ignore_tags)

        # polygons and  polygons_ignore reassignment.
        polygons = results['gt_masks'].masks
        polygons_ignore = results['gt_masks_ignore'].masks

        gt_shrink_mask = self.generate_effective_mask((h, w), polygons_ignore)

        gt_thr, gt_thr_mask = self.generate_thr_map((h, w), polygons)

        results['mask_fields'].clear()  # rm gt_masks encoded by polygons
        results.pop('gt_labels', None)
        results.pop('gt_masks', None)
        results.pop('gt_bboxes', None)
        results.pop('gt_bboxes_ignore', None)

        mapping = {
            'gt_shrink': gt_shrink,
            'gt_shrink_mask': gt_shrink_mask,
            'gt_thr': gt_thr,
            'gt_thr_mask': gt_thr_mask
        }
        for key, value in mapping.items():
            value = value if isinstance(value, list) else [value]
            results[key] = BitmapMasks(value, h, w)
            results['mask_fields'].append(key)

        return results

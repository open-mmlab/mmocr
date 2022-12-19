# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, Tuple

import cv2
import mmcv
import numpy as np
from mmcv.transforms import Resize as MMCV_Resize
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.utils import avoid_cache_randomness, cache_randomness

from mmocr.registry import TRANSFORMS
from mmocr.utils import (bbox2poly, crop_polygon, is_poly_inside_rect,
                         poly2bbox, remove_pipeline_elements, rescale_polygon)
from .wrappers import ImgAugWrapper


@TRANSFORMS.register_module()
@avoid_cache_randomness
class RandomCrop(BaseTransform):
    """Randomly crop images and make sure to contain at least one intact
    instance.

    Required Keys:

    - img
    - gt_polygons
    - gt_bboxes
    - gt_bboxes_labels
    - gt_ignored
    - gt_texts (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_polygons
    - gt_bboxes
    - gt_bboxes_labels
    - gt_ignored
    - gt_texts (optional)

    Args:
        min_side_ratio (float): The ratio of the shortest edge of the cropped
            image to the original image size.
    """

    def __init__(self, min_side_ratio: float = 0.4) -> None:
        if not 0. <= min_side_ratio <= 1.:
            raise ValueError('`min_side_ratio` should be in range [0, 1],')
        self.min_side_ratio = min_side_ratio

    def _sample_valid_start_end(self, valid_array: np.ndarray, min_len: int,
                                max_start_idx: int,
                                min_end_idx: int) -> Tuple[int, int]:
        """Sample a start and end idx on a given axis that contains at least
        one polygon. There should be at least one intact polygon bounded by
        max_start_idx and min_end_idx.

        Args:
            valid_array (ndarray): A 0-1 mask 1D array indicating valid regions
                on the axis. 0 indicates text regions which are not allowed to
                be sampled from.
            min_len (int): Minimum distance between two start and end points.
            max_start_idx (int): The maximum start index.
            min_end_idx (int): The minimum end index.

        Returns:
            tuple(int, int): Start and end index on a given axis, where
            0 <= start < max_start_idx and
            min_end_idx <= end < len(valid_array).
        """
        assert isinstance(min_len, int)
        assert len(valid_array) > min_len

        start_array = valid_array.copy()
        max_start_idx = min(len(start_array) - min_len, max_start_idx)
        start_array[max_start_idx:] = 0
        start_array[0] = 1
        diff_array = np.hstack([0, start_array]) - np.hstack([start_array, 0])
        region_starts = np.where(diff_array < 0)[0]
        region_ends = np.where(diff_array > 0)[0]
        region_ind = np.random.randint(0, len(region_starts))
        start = np.random.randint(region_starts[region_ind],
                                  region_ends[region_ind])

        end_array = valid_array.copy()
        min_end_idx = max(start + min_len, min_end_idx)
        end_array[:min_end_idx] = 0
        end_array[-1] = 1
        diff_array = np.hstack([0, end_array]) - np.hstack([end_array, 0])
        region_starts = np.where(diff_array < 0)[0]
        region_ends = np.where(diff_array > 0)[0]
        region_ind = np.random.randint(0, len(region_starts))
        # Note that end index will never be region_ends[region_ind]
        # and therefore end index is always in range [0, w+1]
        end = np.random.randint(region_starts[region_ind],
                                region_ends[region_ind])
        return start, end

    def _sample_crop_box(self, img_size: Tuple[int, int],
                         results: Dict) -> np.ndarray:
        """Generate crop box which only contains intact polygon instances with
        the number >= 1.

        Args:
            img_size (tuple(int, int)): The image size (h, w).
            results (dict): The results dict.

        Returns:
            ndarray: Crop area in shape (4, ).
        """
        assert isinstance(img_size, tuple)
        h, w = img_size[:2]

        # Crop box can be represented by any integer numbers in
        # range [0, w] and [0, h]
        x_valid_array = np.ones(w + 1, dtype=np.int32)
        y_valid_array = np.ones(h + 1, dtype=np.int32)

        polygons = results['gt_polygons']

        # Randomly select a polygon that must be inside
        # the cropped region
        kept_poly_idx = np.random.randint(0, len(polygons))
        for i, polygon in enumerate(polygons):
            polygon = polygon.reshape((-1, 2))

            clip_x = np.clip(polygon[:, 0], 0, w)
            clip_y = np.clip(polygon[:, 1], 0, h)
            min_x = np.floor(np.min(clip_x)).astype(np.int32)
            min_y = np.floor(np.min(clip_y)).astype(np.int32)
            max_x = np.ceil(np.max(clip_x)).astype(np.int32)
            max_y = np.ceil(np.max(clip_y)).astype(np.int32)

            x_valid_array[min_x:max_x] = 0
            y_valid_array[min_y:max_y] = 0

            if i == kept_poly_idx:
                max_x_start = min_x
                min_x_end = max_x
                max_y_start = min_y
                min_y_end = max_y

        min_w = int(w * self.min_side_ratio)
        min_h = int(h * self.min_side_ratio)

        x1, x2 = self._sample_valid_start_end(x_valid_array, min_w,
                                              max_x_start, min_x_end)
        y1, y2 = self._sample_valid_start_end(y_valid_array, min_h,
                                              max_y_start, min_y_end)

        return np.array([x1, y1, x2, y2])

    def _crop_img(self, img: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Crop image given a bbox region.
            Args:
                img (ndarray): Image.
                bbox (ndarray): Cropping region in shape (4, )

            Returns:
                ndarray: Cropped image.
        """
        assert img.ndim == 3
        h, w, _ = img.shape
        assert 0 <= bbox[1] < bbox[3] <= h
        assert 0 <= bbox[0] < bbox[2] <= w
        return img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    def transform(self, results: Dict) -> Dict:
        """Applying random crop on results.
        Args:
            results (dict): Result dict contains the data to transform.

        Returns:
            dict: The transformed data.
        """
        if len(results['gt_polygons']) < 1:
            return results

        crop_box = self._sample_crop_box(results['img'].shape, results)
        img = self._crop_img(results['img'], crop_box)
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        crop_x = crop_box[0]
        crop_y = crop_box[1]
        crop_w = crop_box[2] - crop_box[0]
        crop_h = crop_box[3] - crop_box[1]

        labels = results['gt_bboxes_labels']
        valid_labels = []
        ignored = results['gt_ignored']
        valid_ignored = []
        if 'gt_texts' in results:
            valid_texts = []
            texts = results['gt_texts']

        polys = results['gt_polygons']
        valid_polys = []
        for idx, poly in enumerate(polys):
            poly = poly.reshape(-1, 2)
            poly = (poly - (crop_x, crop_y)).flatten()
            if is_poly_inside_rect(poly, [0, 0, crop_w, crop_h]):
                valid_polys.append(poly)
                valid_labels.append(labels[idx])
                valid_ignored.append(ignored[idx])
                if 'gt_texts' in results:
                    valid_texts.append(texts[idx])
        results['gt_polygons'] = valid_polys
        results['gt_bboxes_labels'] = np.array(valid_labels, dtype=np.int64)
        results['gt_ignored'] = np.array(valid_ignored, dtype=bool)
        if 'gt_texts' in results:
            results['gt_texts'] = valid_texts
        valid_bboxes = [poly2bbox(poly) for poly in results['gt_polygons']]
        results['gt_bboxes'] = np.array(valid_bboxes).astype(
            np.float32).reshape(-1, 4)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(min_side_ratio = {self.min_side_ratio})'
        return repr_str


@TRANSFORMS.register_module()
class RandomRotate(BaseTransform):
    """Randomly rotate the image, boxes, and polygons. For recognition task,
    only the image will be rotated. If set ``use_canvas`` as True, the shape of
    rotated image might be modified based on the rotated angle size, otherwise,
    the image will keep the shape before rotation.

    Required Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_polygons (optional)

    Modified Keys:

    - img
    - img_shape (optional)
    - gt_bboxes (optional)
    - gt_polygons (optional)

    Added Keys:

    - rotated_angle

    Args:
        max_angle (int): The maximum rotation angle (can be bigger than 180 or
            a negative). Defaults to 10.
        pad_with_fixed_color (bool): The flag for whether to pad rotated
            image with fixed value. Defaults to False.
        pad_value (tuple[int, int, int]): The color value for padding rotated
            image. Defaults to (0, 0, 0).
        use_canvas (bool): Whether to create a canvas for rotated image.
            Defaults to False. If set true, the image shape may be modified.
    """

    def __init__(
        self,
        max_angle: int = 10,
        pad_with_fixed_color: bool = False,
        pad_value: Tuple[int, int, int] = (0, 0, 0),
        use_canvas: bool = False,
    ) -> None:
        if not isinstance(max_angle, int):
            raise TypeError('`max_angle` should be an integer'
                            f', but got {type(max_angle)} instead')
        if not isinstance(pad_with_fixed_color, bool):
            raise TypeError('`pad_with_fixed_color` should be a bool, '
                            f'but got {type(pad_with_fixed_color)} instead')
        if not isinstance(pad_value, (list, tuple)):
            raise TypeError('`pad_value` should be a list or tuple, '
                            f'but got {type(pad_value)} instead')
        if len(pad_value) != 3:
            raise ValueError('`pad_value` should contain three integers')
        if not isinstance(pad_value[0], int) or not isinstance(
                pad_value[1], int) or not isinstance(pad_value[2], int):
            raise ValueError('`pad_value` should contain three integers')

        self.max_angle = max_angle
        self.pad_with_fixed_color = pad_with_fixed_color
        self.pad_value = pad_value
        self.use_canvas = use_canvas

    @cache_randomness
    def _sample_angle(self, max_angle: int) -> float:
        """Sampling a random angle for rotation.

        Args:
            max_angle (int): Maximum rotation angle

        Returns:
            float: The random angle used for rotation
        """
        angle = np.random.random_sample() * 2 * max_angle - max_angle
        return angle

    @staticmethod
    def _cal_canvas_size(ori_size: Tuple[int, int],
                         degree: int) -> Tuple[int, int]:
        """Calculate the canvas size.

        Args:
            ori_size (Tuple[int, int]): The original image size (height, width)
            degree (int): The rotation angle

        Returns:
            Tuple[int, int]: The size of the canvas
        """
        assert isinstance(ori_size, tuple)
        angle = degree * math.pi / 180.0
        h, w = ori_size[:2]

        cos = math.cos(angle)
        sin = math.sin(angle)
        canvas_h = int(w * math.fabs(sin) + h * math.fabs(cos))
        canvas_w = int(w * math.fabs(cos) + h * math.fabs(sin))

        canvas_size = (canvas_h, canvas_w)
        return canvas_size

    @staticmethod
    def _rotate_points(center: Tuple[float, float],
                       points: np.array,
                       theta: float,
                       center_shift: Tuple[int, int] = (0, 0)) -> np.array:
        """Rotating a set of points according to the given theta.

        Args:
            center (Tuple[float, float]): The coordinate of the canvas center
            points (np.array): A set of points needed to be rotated
            theta (float): Rotation angle
            center_shift (Tuple[int, int]): The shifting offset of the center
                coordinate

        Returns:
            np.array: The rotated coordinates of the input points
        """
        (center_x, center_y) = center
        center_y = -center_y
        x, y = points[::2], points[1::2]
        y = -y

        theta = theta / 180 * math.pi
        cos = math.cos(theta)
        sin = math.sin(theta)

        x = (x - center_x)
        y = (y - center_y)

        _x = center_x + x * cos - y * sin + center_shift[0]
        _y = -(center_y + x * sin + y * cos) + center_shift[1]

        points[::2], points[1::2] = _x, _y
        return points

    def _rotate_img(self, results: Dict) -> Tuple[int, int]:
        """Rotating the input image based on the given angle.

        Args:
            results (dict): Result dict containing the data to transform.

        Returns:
            Tuple[int, int]: The shifting offset of the center point.
        """
        if results.get('img', None) is not None:
            h = results['img'].shape[0]
            w = results['img'].shape[1]
            rotation_matrix = cv2.getRotationMatrix2D(
                (w / 2, h / 2), results['rotated_angle'], 1)

            canvas_size = self._cal_canvas_size((h, w),
                                                results['rotated_angle'])
            center_shift = (int(
                (canvas_size[1] - w) / 2), int((canvas_size[0] - h) / 2))
            rotation_matrix[0, 2] += int((canvas_size[1] - w) / 2)
            rotation_matrix[1, 2] += int((canvas_size[0] - h) / 2)
            if self.pad_with_fixed_color:
                rotated_img = cv2.warpAffine(
                    results['img'],
                    rotation_matrix, (canvas_size[1], canvas_size[0]),
                    flags=cv2.INTER_NEAREST,
                    borderValue=self.pad_value)
            else:
                mask = np.zeros_like(results['img'])
                (h_ind, w_ind) = (np.random.randint(0, h * 7 // 8),
                                  np.random.randint(0, w * 7 // 8))
                img_cut = results['img'][h_ind:(h_ind + h // 9),
                                         w_ind:(w_ind + w // 9)]
                img_cut = mmcv.imresize(img_cut,
                                        (canvas_size[1], canvas_size[0]))
                mask = cv2.warpAffine(
                    mask,
                    rotation_matrix, (canvas_size[1], canvas_size[0]),
                    borderValue=[1, 1, 1])
                rotated_img = cv2.warpAffine(
                    results['img'],
                    rotation_matrix, (canvas_size[1], canvas_size[0]),
                    borderValue=[0, 0, 0])
                rotated_img = rotated_img + img_cut * mask

            results['img'] = rotated_img
        else:
            raise ValueError('`img` is not found in results')

        return center_shift

    def _rotate_bboxes(self, results: Dict, center_shift: Tuple[int,
                                                                int]) -> None:
        """Rotating the bounding boxes based on the given angle.

        Args:
            results (dict): Result dict containing the data to transform.
            center_shift (Tuple[int, int]): The shifting offset of the
                center point
        """
        if results.get('gt_bboxes', None) is not None:
            height, width = results['img_shape']
            box_list = []
            for box in results['gt_bboxes']:
                rotated_box = self._rotate_points((width / 2, height / 2),
                                                  bbox2poly(box),
                                                  results['rotated_angle'],
                                                  center_shift)
                rotated_box = poly2bbox(rotated_box)
                box_list.append(rotated_box)

            results['gt_bboxes'] = np.array(
                box_list, dtype=np.float32).reshape(-1, 4)

    def _rotate_polygons(self, results: Dict,
                         center_shift: Tuple[int, int]) -> None:
        """Rotating the polygons based on the given angle.

        Args:
            results (dict): Result dict containing the data to transform.
            center_shift (Tuple[int, int]): The shifting offset of the
                center point
        """
        if results.get('gt_polygons', None) is not None:
            height, width = results['img_shape']
            polygon_list = []
            for poly in results['gt_polygons']:
                rotated_poly = self._rotate_points(
                    (width / 2, height / 2), poly, results['rotated_angle'],
                    center_shift)
                polygon_list.append(rotated_poly)
            results['gt_polygons'] = polygon_list

    def transform(self, results: Dict) -> Dict:
        """Applying random rotate on results.

        Args:
            results (Dict): Result dict containing the data to transform.
            center_shift (Tuple[int, int]): The shifting offset of the
                center point

        Returns:
            dict: The transformed data
        """
        # TODO rotate char_quads & char_rects for SegOCR
        if self.use_canvas:
            results['rotated_angle'] = self._sample_angle(self.max_angle)
            # rotate image
            center_shift = self._rotate_img(results)
            # rotate gt_bboxes
            self._rotate_bboxes(results, center_shift)
            # rotate gt_polygons
            self._rotate_polygons(results, center_shift)

            results['img_shape'] = (results['img'].shape[0],
                                    results['img'].shape[1])
        else:
            args = [
                dict(
                    cls='Affine',
                    rotate=[-self.max_angle, self.max_angle],
                    backend='cv2',
                    order=0)  # order=0 -> cv2.INTER_NEAREST
            ]
            imgaug_transform = ImgAugWrapper(args)
            results = imgaug_transform(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(max_angle = {self.max_angle}'
        repr_str += f', pad_with_fixed_color = {self.pad_with_fixed_color}'
        repr_str += f', pad_value = {self.pad_value}'
        repr_str += f', use_canvas = {self.use_canvas})'
        return repr_str


@TRANSFORMS.register_module()
class Resize(MMCV_Resize):
    """Resize image & bboxes & polygons.

    This transform resizes the input image according to ``scale`` or
    ``scale_factor``. Bboxes and polygons are then resized with the same
    scale factor. if ``scale`` and ``scale_factor`` are both set, it will use
    ``scale`` to resize.

    Required Keys:

    - img
    - img_shape
    - gt_bboxes
    - gt_polygons


    Modified Keys:

    - img
    - img_shape
    - gt_bboxes
    - gt_polygons

    Added Keys:

    - scale
    - scale_factor
    - keep_ratio

    Args:
        scale (int or tuple): Image scales for resizing. Defaults to None.
        scale_factor (float or tuple[float, float]): Scale factors for
            resizing. It's either a factor applicable to both dimensions or
            in the form of (scale_w, scale_h). Defaults to None.
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Defaults to False.
        clip_object_border (bool): Whether to clip the objects outside the
            border of the image. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'bilinear'.
    """

    def _resize_img(self, results: dict) -> None:
        """Resize images with ``results['scale']``.

        If no image is provided, only resize ``results['img_shape']``.
        """
        if results.get('img', None) is not None:
            return super()._resize_img(results)
        h, w = results['img_shape']
        if self.keep_ratio:
            new_w, new_h = mmcv.rescale_size((w, h),
                                             results['scale'],
                                             return_scale=False)
        else:
            new_w, new_h = results['scale']
        w_scale = new_w / w
        h_scale = new_h / h
        results['img_shape'] = (new_h, new_w)
        results['scale'] = (new_w, new_h)
        results['scale_factor'] = (w_scale, h_scale)
        results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results: dict) -> None:
        """Resize bounding boxes."""
        super()._resize_bboxes(results)
        if results.get('gt_bboxes', None) is not None:
            results['gt_bboxes'] = results['gt_bboxes'].astype(np.float32)

    def _resize_polygons(self, results: dict) -> None:
        """Resize polygons with ``results['scale_factor']``."""
        if results.get('gt_polygons', None) is not None:
            polygons = results['gt_polygons']
            polygons_resize = []
            for idx, polygon in enumerate(polygons):
                polygon = rescale_polygon(polygon, results['scale_factor'])
                if self.clip_object_border:
                    crop_bbox = np.array([
                        0, 0, results['img_shape'][1], results['img_shape'][0]
                    ])
                    polygon = crop_polygon(polygon, crop_bbox)
                if polygon is not None:
                    polygons_resize.append(polygon.astype(np.float32))
                else:
                    polygons_resize.append(
                        np.zeros_like(polygons[idx], dtype=np.float32))
            results['gt_polygons'] = polygons_resize

    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes and polygons.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_polygons',
            'scale', 'scale_factor', 'height', 'width', and 'keep_ratio' keys
            are updated in result dict.
        """
        results = super().transform(results)
        self._resize_polygons(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(scale={self.scale}, '
        repr_str += f'scale_factor={self.scale_factor}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'clip_object_border={self.clip_object_border}), '
        repr_str += f'backend={self.backend}), '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str


@TRANSFORMS.register_module()
class RemoveIgnored(BaseTransform):
    """Removed ignored elements from the pipeline.

    Required Keys:

    - gt_ignored
    - gt_polygons (optional)
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_texts (optional)

    Modified Keys:

    - gt_ignored
    - gt_polygons (optional)
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_texts (optional)
    """

    def transform(self, results: Dict) -> Dict:
        remove_inds = np.where(results['gt_ignored'])[0]
        if len(remove_inds) == len(results['gt_ignored']):
            return None
        return remove_pipeline_elements(results, remove_inds)

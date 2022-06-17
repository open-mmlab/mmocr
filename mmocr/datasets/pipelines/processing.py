# Copyright (c) OpenMMLab. All rights reserved.
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import mmcv
import numpy as np
from mmcv.transforms import RandomFlip as MMCV_RandomFlip
from mmcv.transforms import Resize as MMCV_Resize
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.utils import avoid_cache_randomness, cache_randomness
from shapely.geometry import Polygon as plg

import mmocr.core.evaluation.utils as eval_utils
from mmocr.registry import TRANSFORMS
from mmocr.utils import (bbox2poly, crop_polygon, is_poly_inside_rect,
                         poly2bbox, rescale_polygon)
from .wrappers import ImgAug


@TRANSFORMS.register_module()
class PyramidRescale(BaseTransform):
    """Resize the image to the base shape, downsample it with gaussian pyramid,
    and rescale it back to original size.

    Adapted from https://github.com/FangShancheng/ABINet.

    Required Keys:

    - img (ndarray)

    Modified Keys:

    - img (ndarray)

    Args:
        factor (int): The decay factor from base size, or the number of
            downsampling operations from the base layer.
        base_shape (tuple[int, int]): The shape (width, height) of the base
            layer of the pyramid.
        randomize_factor (bool): If True, the final factor would be a random
            integer in [0, factor].
    """

    def __init__(self,
                 factor: int = 4,
                 base_shape: Tuple[int, int] = (128, 512),
                 randomize_factor: bool = True) -> None:
        if not isinstance(factor, int):
            raise TypeError('`factor` should be an integer, '
                            f'but got {type(factor)} instead')
        if not isinstance(base_shape, (list, tuple)):
            raise TypeError('`base_shape` should be a list or tuple, '
                            f'but got {type(base_shape)} instead')
        if not len(base_shape) == 2:
            raise ValueError('`base_shape` should contain two integers')
        if not isinstance(base_shape[0], int) or not isinstance(
                base_shape[1], int):
            raise ValueError('`base_shape` should contain two integers')
        if not isinstance(randomize_factor, bool):
            raise TypeError('`randomize_factor` should be a bool, '
                            f'but got {type(randomize_factor)} instead')

        self.factor = factor
        self.randomize_factor = randomize_factor
        self.base_w, self.base_h = base_shape

    @cache_randomness
    def get_random_factor(self) -> float:
        """Get the randomized factor.

        Returns:
            float: The randomized factor.
        """
        return np.random.randint(0, self.factor + 1)

    def transform(self, results: Dict) -> Dict:
        """Applying pyramid rescale on results.

        Args:
            results (dict): Result dict containing the data to transform.

        Returns:
            Dict: The transformed data.
        """

        assert 'img' in results, '`img` is not found in results'
        if self.randomize_factor:
            self.factor = self.get_random_factor()
        if self.factor == 0:
            return results
        img = results['img']
        src_h, src_w = img.shape[:2]
        scale_img = mmcv.imresize(img, (self.base_w, self.base_h))
        for _ in range(self.factor):
            scale_img = cv2.pyrDown(scale_img)
        scale_img = mmcv.imresize(scale_img, (src_w, src_h))
        results['img'] = scale_img
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(factor = {self.factor}'
        repr_str += f', randomize_factor = {self.randomize_factor}'
        repr_str += f', base_w = {self.base_w}'
        repr_str += f', base_h = {self.base_h})'
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

            results['gt_bboxes'] = np.array(box_list)

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
            imgaug_transform = ImgAug(args)
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
class TextDetRandomCropFlip(BaseTransform):
    # TODO Rename this transformer; Refactor the redundant code.
    """Random crop and flip a patch in the image. Only used in text detection
    task.

    Required Keys:

    - img
    - gt_bboxes
    - gt_polygons

    Modified Keys:

    - img
    - gt_bboxes
    - gt_polygons

    Args:
        pad_ratio (float): The ratio of padding. Defaults to 0.1.
        crop_ratio (float): The ratio of cropping. Defaults to 0.5.
        iter_num (int): Number of operations. Defaults to 1.
        min_area_ratio (float): Minimal area ratio between cropped patch
            and original image. Defaults to 0.2.
        epsilon (float): The threshold of polygon IoU between cropped area
            and polygon, which is used to avoid cropping text instances.
            Defaults to 0.01.
    """

    def __init__(self,
                 pad_ratio: float = 0.1,
                 crop_ratio: float = 0.5,
                 iter_num: int = 1,
                 min_area_ratio: float = 0.2,
                 epsilon: float = 1e-2) -> None:
        if not isinstance(pad_ratio, float):
            raise TypeError('`pad_ratio` should be an float, '
                            f'but got {type(pad_ratio)} instead')
        if not isinstance(crop_ratio, float):
            raise TypeError('`crop_ratio` should be a float, '
                            f'but got {type(crop_ratio)} instead')
        if not isinstance(iter_num, int):
            raise TypeError('`iter_num` should be an integer, '
                            f'but got {type(iter_num)} instead')
        if not isinstance(min_area_ratio, float):
            raise TypeError('`min_area_ratio` should be a float, '
                            f'but got {type(min_area_ratio)} instead')
        if not isinstance(epsilon, float):
            raise TypeError('`epsilon` should be a float, '
                            f'but got {type(epsilon)} instead')

        self.pad_ratio = pad_ratio
        self.epsilon = epsilon
        self.crop_ratio = crop_ratio
        self.iter_num = iter_num
        self.min_area_ratio = min_area_ratio

    @cache_randomness
    def _random_prob(self) -> float:
        """Get the random prob to decide whether apply the transform.

        Returns:
            float: The probability
        """
        return random.random()

    @cache_randomness
    def _random_flip_type(self) -> int:
        """Get the random flip type.

        Returns:
            int: The flip type index. (0: horizontal; 1: vertical; 2: both)
        """
        return np.random.randint(3)

    @cache_randomness
    def _random_choice(self, axis: np.ndarray) -> np.ndarray:
        """Randomly select two coordinates from the axis.

        Args:
            axis (np.ndarray): Result dict containing the data to transform

        Returns:
            np.ndarray: The selected coordinates
        """
        return np.random.choice(axis, size=2)

    def transform(self, results: Dict) -> Dict:
        """Applying random crop flip on results.

        Args:
            results (dict): Result dict containing the data to transform

        Returns:
            dict: The transformed data
        """
        assert 'img' in results, '`img` is not found in results'
        for _ in range(self.iter_num):
            results = self._random_crop_flip_polygons(results)
        bboxes = [poly2bbox(poly) for poly in results['gt_polygons']]
        results['gt_bboxes'] = np.array(
            bboxes, dtype=np.float32).reshape(-1, 4)
        return results

    def _random_crop_flip_polygons(self, results: Dict) -> Dict:
        """Applying random crop flip on polygons.

        Args:
            results (dict): Result dict containing the data to transform

        Returns:
            dict: The transformed data
        """
        if results.get('gt_polygons', None) is None:
            return results

        image = results['img']
        polygons = results['gt_polygons']
        if len(polygons) == 0 or self._random_prob() > self.crop_ratio:
            return results

        h, w = results['img_shape']
        area = h * w
        pad_h = int(h * self.pad_ratio)
        pad_w = int(w * self.pad_ratio)
        h_axis, w_axis = self._generate_crop_target(image, polygons, pad_h,
                                                    pad_w)
        if len(h_axis) == 0 or len(w_axis) == 0:
            return results

        # At most 10 attempts
        for _ in range(10):
            polys_keep = []
            polys_new = []
            kept_idxs = []
            xx = self._random_choice(w_axis)
            yy = self._random_choice(h_axis)
            xmin = np.clip(np.min(xx) - pad_w, 0, w - 1)
            xmax = np.clip(np.max(xx) - pad_w, 0, w - 1)
            ymin = np.clip(np.min(yy) - pad_h, 0, h - 1)
            ymax = np.clip(np.max(yy) - pad_h, 0, h - 1)
            if (xmax - xmin) * (ymax - ymin) < area * self.min_area_ratio:
                # Skip when cropped area is too small
                continue

            pts = np.stack([[xmin, xmax, xmax, xmin],
                            [ymin, ymin, ymax, ymax]]).T.astype(np.int32)
            pp = plg(pts)
            success_flag = True
            for poly_idx, polygon in enumerate(polygons):
                ppi = plg(polygon.reshape(-1, 2))
                # TODO Move this eval_utils to point_utils?
                ppiou = eval_utils.poly_intersection(ppi, pp)
                if np.abs(ppiou - float(ppi.area)) > self.epsilon and \
                        np.abs(ppiou) > self.epsilon:
                    success_flag = False
                    break
                kept_idxs.append(poly_idx)
                if np.abs(ppiou - float(ppi.area)) < self.epsilon:
                    polys_new.append(polygon)
                else:
                    polys_keep.append(polygon)

            if success_flag:
                break

        cropped = image[ymin:ymax, xmin:xmax, :]
        select_type = self._random_flip_type()
        if select_type == 0:
            img = np.ascontiguousarray(cropped[:, ::-1])
        elif select_type == 1:
            img = np.ascontiguousarray(cropped[::-1, :])
        else:
            img = np.ascontiguousarray(cropped[::-1, ::-1])
        image[ymin:ymax, xmin:xmax, :] = img
        results['img'] = image

        if len(polys_new) != 0:
            height, width, _ = cropped.shape
            if select_type == 0:
                for idx, polygon in enumerate(polys_new):
                    poly = polygon.reshape(-1, 2)
                    poly[:, 0] = width - poly[:, 0] + 2 * xmin
                    polys_new[idx] = poly.reshape(-1, )
            elif select_type == 1:
                for idx, polygon in enumerate(polys_new):
                    poly = polygon.reshape(-1, 2)
                    poly[:, 1] = height - poly[:, 1] + 2 * ymin
                    polys_new[idx] = poly.reshape(-1, )
            else:
                for idx, polygon in enumerate(polys_new):
                    poly = polygon.reshape(-1, 2)
                    poly[:, 0] = width - poly[:, 0] + 2 * xmin
                    poly[:, 1] = height - poly[:, 1] + 2 * ymin
                    polys_new[idx] = poly.reshape(-1, )
            polygons = polys_keep + polys_new
            # ignored = polys_keep_ignore_idx + polys_new_ignore_idx
            results['gt_polygons'] = polygons
            results['gt_ignored'] = results['gt_ignored'][kept_idxs]
            results['gt_bboxes_labels'] = results['gt_bboxes_labels'][
                kept_idxs]
        return results

    def _generate_crop_target(self, image: np.ndarray,
                              all_polys: List[np.ndarray], pad_h: int,
                              pad_w: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate cropping target and make sure not to crop the polygon
        instances.

        Args:
            image (np.ndarray): The image waited to be crop.
            all_polys (list[np.ndarray]): Ground-truth polygons.
            pad_h (int): Padding length of height.
            pad_w (int): Padding length of width.

        Returns:
            (np.ndarray, np.ndarray): Returns a tuple ``(h_axis, w_axis)``,
            where ``h_axis`` is the vertical cropping range and ``w_axis``
            is the horizontal cropping range.
        """
        h, w, _ = image.shape
        h_array = np.zeros((h + pad_h * 2), dtype=np.int32)
        w_array = np.zeros((w + pad_w * 2), dtype=np.int32)

        text_polys = []
        for polygon in all_polys:
            rect = cv2.minAreaRect(polygon.astype(np.int32).reshape(-1, 2))
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            text_polys.append([box[0], box[1], box[2], box[3]])

        polys = np.array(text_polys, dtype=np.int32)
        for poly in polys:
            poly = np.round(poly, decimals=0).astype(np.int32)
            minx, maxx = np.min(poly[:, 0]), np.max(poly[:, 0])
            miny, maxy = np.min(poly[:, 1]), np.max(poly[:, 1])
            w_array[minx + pad_w:maxx + pad_w] = 1
            h_array[miny + pad_h:maxy + pad_h] = 1

        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]
        return h_axis, w_axis

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(pad_ratio = {self.pad_ratio}'
        repr_str += f', crop_ratio = {self.crop_ratio}'
        repr_str += f', iter_num = {self.iter_num}'
        repr_str += f', min_area_ratio = {self.min_area_ratio}'
        repr_str += f', epsilon = {self.epsilon})'
        return repr_str


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
        results['gt_bboxes'] = np.array(valid_bboxes).astype(np.float32)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(min_side_ratio = {self.min_side_ratio})'
        return repr_str


@TRANSFORMS.register_module()
@avoid_cache_randomness
class TextDetRandomCrop(BaseTransform):
    """Randomly select a region and crop images to a target size and make sure
    to contain text region. This transform may break up text instances, and for
    broken text instances, we will crop it's bbox and polygon coordinates. This
    transform is recommend to be used in segmentation-based network.

    Required Keys:

    - img
    - gt_polygons
    - gt_bboxes
    - gt_bboxes_labels
    - gt_ignored

    Modified Keys:

    - img
    - img_shape
    - gt_polygons
    - gt_bboxes
    - gt_bboxes_labels
    - gt_ignored

    Args:
        target_size (tuple(int, int) or int): Target size for the cropped
            image. If it's a tuple, then target width and target height will be
            ``target_size[0]`` and ``target_size[1]``, respectively. If it's an
            integer, them both target width and target height will be
            ``target_size``.
        positive_sample_ratio (float): The probability of sampling regions
            that go through text regions. Defaults to 5. / 8.
    """

    def __init__(self,
                 target_size: Tuple[int, int] or int,
                 positive_sample_ratio: float = 5.0 / 8.0) -> None:
        self.target_size = target_size if isinstance(
            target_size, tuple) else (target_size, target_size)
        self.positive_sample_ratio = positive_sample_ratio

    def _get_postive_prob(self) -> float:
        """Get the probability to do positive sample.

        Returns:
            float: The probability to do positive sample.
        """
        return np.random.random_sample()

    def _sample_num(self, start, end):
        """Sample a number in range [start, end].

        Args:
            start (int): Starting point.
            end (int): Ending point.

        Returns:
            (int): Sampled number.
        """
        return random.randint(start, end)

    def _sample_offset(self, gt_polygons: Sequence[np.ndarray],
                       img_size: Tuple[int, int]) -> Tuple[int, int]:
        """Samples the top-left coordinate of a crop region, ensuring that the
        cropped region contains at least one polygon.

        Args:
            gt_polygons (list(ndarray)) : Polygons.
            img_size (tuple(int, int)) : Image size in the format of
                (height, width).

        Returns:
            tuple(int, int): Top-left coordinate of the cropped region.
        """
        h, w = img_size
        t_w, t_h = self.target_size

        # target size is bigger than origin size
        t_h = t_h if t_h < h else h
        t_w = t_w if t_w < w else w
        if (gt_polygons is not None and len(gt_polygons) > 0
                and self._get_postive_prob() < self.positive_sample_ratio):

            # make sure to crop the positive region

            # the minimum top left to crop positive region (h,w)
            tl = np.array([h + 1, w + 1], dtype=np.int32)
            for gt_polygon in gt_polygons:
                temp_point = np.min(gt_polygon.reshape(2, -1), axis=1)
                if temp_point[0] <= tl[0]:
                    tl[0] = temp_point[0]
                if temp_point[1] <= tl[1]:
                    tl[1] = temp_point[1]
            tl = tl - (t_h, t_w)
            tl[tl < 0] = 0
            # the maximum bottum right to crop positive region
            br = np.array([0, 0], dtype=np.int32)
            for gt_polygon in gt_polygons:
                temp_point = np.max(gt_polygon.reshape(2, -1), axis=1)
                if temp_point[0] > br[0]:
                    br[0] = temp_point[0]
                if temp_point[1] > br[1]:
                    br[1] = temp_point[1]
            br = br - (t_h, t_w)
            br[br < 0] = 0

            # if br is too big so that crop the outside region of img
            br[0] = min(br[0], h - t_h)
            br[1] = min(br[1], w - t_w)
            #
            h = self._sample_num(tl[0], br[0]) if tl[0] < br[0] else 0
            w = self._sample_num(tl[1], br[1]) if tl[1] < br[1] else 0
        else:
            # make sure not to crop outside of img

            h = self._sample_num(0, h - t_h) if h - t_h > 0 else 0
            w = self._sample_num(0, w - t_w) if w - t_w > 0 else 0

        return (h, w)

    def _crop_img(self, img: np.ndarray, offset: Tuple[int, int],
                  target_size: Tuple[int, int]) -> np.ndarray:
        """Crop the image given an offset and a target size.

        Args:
            img (ndarray): Image.
            offset (Tuple[int. int]): Coordinates of the starting point.
            target_size: Target image size.
        """
        h, w = img.shape[:2]
        target_size = target_size[::-1]
        br = np.min(
            np.stack((np.array(offset) + np.array(target_size), np.array(
                (h, w)))),
            axis=0)
        return img[offset[0]:br[0], offset[1]:br[1]], np.array(
            [offset[1], offset[0], br[1], br[0]])

    def _crop_polygons(self, polygons: Sequence[np.ndarray],
                       crop_bbox: np.ndarray) -> Sequence[np.ndarray]:
        """Crop polygons to be within a crop region. If polygon crosses the
        crop_bbox, we will keep the part left in crop_bbox by cropping its
        boardline.

        Args:
            polygons (list(ndarray)): List of polygons [(N1, ), (N2, ), ...].
            crop_bbox (ndarray): Cropping region. [x1, y1, x2, y1].

        Returns
            tuple(List(ArrayLike), list[int]):
                - (List(ArrayLike)): The rest of the polygons located in the
                    crop region.
                - (list[int]): Index list of the reserved polygons.
        """
        polygons_cropped = []
        kept_idx = []
        for idx, polygon in enumerate(polygons):
            if polygon.size < 6:
                continue
            poly = crop_polygon(polygon, crop_bbox)
            if poly is not None:
                poly = poly.reshape(-1, 2) - (crop_bbox[0], crop_bbox[1])
                polygons_cropped.append(poly.reshape(-1))
                kept_idx.append(idx)
        return (polygons_cropped, kept_idx)

    def transform(self, results: Dict) -> Dict:
        """Applying random crop on results.
        Args:
            results (dict): Result dict contains the data to transform.

        Returns:
            dict: The transformed data
        """
        if self.target_size == results['img'].shape[:2][::-1]:
            return results
        gt_polygons = results['gt_polygons']
        crop_offset = self._sample_offset(gt_polygons,
                                          results['img'].shape[:2])
        img, crop_bbox = self._crop_img(results['img'], crop_offset,
                                        self.target_size)
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        gt_polygons, polygon_kept_idx = self._crop_polygons(
            gt_polygons, crop_bbox)
        bboxes = [poly2bbox(poly) for poly in gt_polygons]
        results['gt_bboxes'] = np.array(
            bboxes, dtype=np.float32).reshape(-1, 4)

        results['gt_polygons'] = gt_polygons
        results['gt_bboxes_labels'] = results['gt_bboxes_labels'][
            polygon_kept_idx]
        results['gt_ignored'] = results['gt_ignored'][polygon_kept_idx]
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(target_size = {self.target_size}, '
        repr_str += f'positive_sample_ratio = {self.positive_sample_ratio})'
        return repr_str


@TRANSFORMS.register_module()
class RescaleToHeight(BaseTransform):
    """Rescale the image to the height according to setting and keep the aspect
    ratio unchanged if possible. However, if any of ``min_width``,
    ``max_width`` or ``width_divisor`` are specified, aspect ratio may still be
    changed to ensure the width meets these constraints.

    Required Keys:

    - img

    Modified Keys:

    - img
    - img_shape

    Added Keys:

    - scale
    - scale_factor
    - keep_ratio

    Args:
        height (int): Height of rescaled image.
        min_width (int, optional): Minimum width of rescaled image. Defaults
            to None.
        max_width (int, optional): Maximum width of rescaled image. Defaults
            to None.
        width_divisor (int): The divisor of width size. Defaults to 1.
        resize_cfg (dict):  (dict): Config to construct the Resize transform.
            Refer to ``Resize`` for detail. Defaults to
            ``dict(type='Resize')``.
    """

    def __init__(self,
                 height: int,
                 min_width: Optional[int] = None,
                 max_width: Optional[int] = None,
                 width_divisor: int = 1,
                 resize_cfg: dict = dict(type='Resize')) -> None:
        super().__init__()
        assert isinstance(height, int)
        assert isinstance(width_divisor, int)
        if min_width is not None:
            assert isinstance(min_width, int)
        if max_width is not None:
            assert isinstance(max_width, int)
        self.width_divisor = width_divisor
        self.height = height
        self.min_width = min_width
        self.max_width = max_width
        self.resize_cfg = resize_cfg
        _resize_cfg = self.resize_cfg.copy()
        _resize_cfg.update(dict(scale=0))
        self.resize = TRANSFORMS.build(_resize_cfg)

    def transform(self, results: Dict) -> Dict:
        """Transform function to resize images, bounding boxes and polygons.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results.
        """
        ori_height, ori_width = results['img'].shape[:2]
        new_width = math.ceil(float(self.height) / ori_height * ori_width)
        if self.min_width is not None:
            new_width = max(self.min_width, new_width)
        if self.max_width is not None:
            new_width = min(self.max_width, new_width)

        if new_width % self.width_divisor != 0:
            new_width = round(
                new_width / self.width_divisor) * self.width_divisor
        # TODO replace up code after testing precision.
        # new_width = math.ceil(
        #     new_width / self.width_divisor) * self.width_divisor
        scale = (new_width, self.height)
        self.resize.scale = scale
        results = self.resize(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(height={self.height}, '
        repr_str += f'min_width={self.min_width}, '
        repr_str += f'max_width={self.max_width}, '
        repr_str += f'width_divisor={self.width_divisor}, '
        repr_str += f'resize_cfg={self.resize_cfg})'
        return repr_str


@TRANSFORMS.register_module()
class PadToWidth(BaseTransform):
    """Only pad the image's width.

    Required Keys:

    - img

    Modified Keys:

    - img
    - img_shape

    Added Keys:

    - pad_shape
    - pad_fixed_size
    - pad_size_divisor
    - valid_ratio

    Args:
        width (int): Target width of padded image. Defaults to None.
        pad_cfg (dict): Config to construct the Resize transform. Refer to
            ``Pad`` for detail. Defaults to ``dict(type='Pad')``.
    """

    def __init__(self, width: int, pad_cfg: dict = dict(type='Pad')) -> None:
        super().__init__()
        assert isinstance(width, int)
        self.width = width
        self.pad_cfg = pad_cfg
        _pad_cfg = self.pad_cfg.copy()
        _pad_cfg.update(dict(size=0))
        self.pad = TRANSFORMS.build(_pad_cfg)

    def transform(self, results: Dict) -> Dict:
        """Call function to pad images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        ori_height, ori_width = results['img'].shape[:2]
        valid_ratio = min(1.0, 1.0 * ori_width / self.width)
        size = (self.width, ori_height)
        self.pad.size = size
        results = self.pad(results)
        results['valid_ratio'] = valid_ratio
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(width={self.width}, '
        repr_str += f'pad_cfg={self.pad_cfg})'
        return repr_str


@TRANSFORMS.register_module()
class SourceImagePad(BaseTransform):
    """Pad Image to target size. It will randomly crop an area from the
    original image and resize it to the target size, then paste the original
    image to its top left corner.

    Required Keys:

    - img

    Modified Keys:

    - img
    - img_shape

    Added Keys:
    - pad_shape
    - pad_fixed_size

    Args:
        target_scale (int or tuple[int, int]]): The target size of padded
            image. If it's an integer, then the padding size would be
            (target_size, target_size). If it's tuple, then ``target_scale[0]``
            should be the width and ``target_scale[1]`` should be the height.
            The size of the padded image will be (target_scale[1],
            target_scale[0])
        crop_ratio (float or Tuple[float, float]): Relative size for the
            crop region. If ``crop_ratio`` is a float, then the initial crop
            size would be
            ``(crop_ratio * img.shape[0], crop_ratio * img.shape[1])`` . If
            ``crop_ratio`` is a tuple, then ``crop_ratio[0]`` is for the width
            and ``crop_ratio[1]`` is for the height. The initial crop size
            would be
            ``(crop_ratio[1] * img.shape[0], crop_ratio[0] * img.shape[1])``.
            Defaults to 1./9.
    """

    def __init__(self,
                 target_scale: Union[int, Tuple[int, int]],
                 crop_ratio: Union[float, Tuple[float,
                                                float]] = 1. / 9) -> None:
        self.target_scale = target_scale if isinstance(
            target_scale, tuple) else (target_scale, target_scale)
        self.crop_ratio = crop_ratio if isinstance(
            crop_ratio, tuple) else (crop_ratio, crop_ratio)

    def transform(self, results: Dict) -> Dict:
        """Pad Image to target size. It will randomly select a small area from
        the  original image and resize it to the target size, then paste the
        original image to its top left corner.

        Args:
            results (Dict): Result dict containing the data to transform.

        Returns:
            (Dict): The transformed data.
        """
        img = results['img']
        h, w = img.shape[:2]
        assert h <= self.target_scale[1] and w <= self.target_scale[
            0], 'image size should be smaller that the target size'
        h_ind = np.random.randint(0, int(h - h * self.crop_ratio[1]) + 1)
        w_ind = np.random.randint(0, int(w - w * self.crop_ratio[0]) + 1)
        img_cut = img[h_ind:int(h_ind + h * self.crop_ratio[1]),
                      w_ind:int(w_ind + w * self.crop_ratio[1])]
        expand_img = mmcv.imresize(img_cut, self.target_scale)
        # paste img to the top left corner of the padding region
        expand_img[0:h, 0:w] = img
        results['img'] = expand_img
        results['img_shape'] = expand_img.shape[:2]
        results['pad_shape'] = expand_img.shape
        results['pad_fixed_size'] = self.target_scale
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(target_scale = {self.target_scale}, '
        repr_str += f'crop_ratio = {self.crop_ratio})'
        return repr_str


@TRANSFORMS.register_module()
class RandomFlip(MMCV_RandomFlip):
    """Flip the image & bbox polygon.

    There are 3 flip modes:

     - ``prob`` is float, ``direction`` is string: the image will be
         ``direction``ly flipped with probability of ``prob`` .
         E.g., ``prob=0.5``, ``direction='horizontal'``,
         then image will be horizontally flipped with probability of 0.5.
     - ``prob`` is float, ``direction`` is list of string: the image will
         be ``direction[i]``ly flipped with probability of
         ``prob/len(direction)``.
         E.g., ``prob=0.5``, ``direction=['horizontal', 'vertical']``,
         then image will be horizontally flipped with probability of 0.25,
         vertically with probability of 0.25.
     - ``prob`` is list of float, ``direction`` is list of string:
         given ``len(prob) == len(direction)``, the image will
         be ``direction[i]``ly flipped with probability of ``prob[i]``.
         E.g., ``prob=[0.3, 0.5]``, ``direction=['horizontal',
         'vertical']``, then image will be horizontally flipped with
         probability of 0.3, vertically with probability of 0.5.

    Required Keys:
        - img
        - gt_bboxes (optional)
        - gt_polygons (optional)

    Modified Keys:
        - img
        - gt_bboxes (optional)
        - gt_polygons (optional)

    Added Keys:
        - flip
        - flip_direction
    Args:
         prob (float | list[float], optional): The flipping probability.
             Defaults to None.
         direction(str | list[str]): The flipping direction. Options
             If input is a list, the length must equal ``prob``. Each
             element in ``prob`` indicates the flip probability of
             corresponding direction. Defaults to 'horizontal'.
    """

    def flip_polygons(self, polygons: Sequence[np.ndarray],
                      img_shape: Tuple[int, int],
                      direction: str) -> Sequence[np.ndarray]:
        """Flip polygons horizontally, vertically or diagonally.

        Args:
            polygons (list[numpy.ndarray): polygons.
            img_shape (tuple[int]): Image shape (height, width)
            direction (str): Flip direction. Options are 'horizontal',
                'vertical' and 'diagonal'.
        Returns:
            list[numpy.ndarray]: Flipped polygons.
        """

        h, w = img_shape
        flipped_polygons = []
        if direction == 'horizontal':
            for polygon in polygons:
                flipped_polygon = polygon.copy()
                flipped_polygon[0::2] = w - polygon[0::2]
                flipped_polygons.append(flipped_polygon)
        elif direction == 'vertical':
            for polygon in polygons:
                flipped_polygon = polygon.copy()
                flipped_polygon[1::2] = h - polygon[1::2]
                flipped_polygons.append(flipped_polygon)
        elif direction == 'diagonal':
            for polygon in polygons:
                flipped_polygon = polygon.copy()
                flipped_polygon[0::2] = w - polygon[0::2]
                flipped_polygon[1::2] = h - polygon[1::2]
                flipped_polygons.append(flipped_polygon)
        else:
            raise ValueError(
                f"Flipping direction must be 'horizontal', 'vertical', \
                  or 'diagnal', but got '{direction}'")
        return flipped_polygons

    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes and polygons."""
        super()._flip(results)
        # flip polygons
        if results.get('gt_polygons', None) is not None:
            results['gt_polygons'] = self.flip_polygons(
                results['gt_polygons'], results['img'].shape[:2],
                results['flip_direction'])


@TRANSFORMS.register_module()
@avoid_cache_randomness
class ShortScaleAspectJitter(BaseTransform):
    """First rescale the image for its shorter side to reach the short_size and
    then jitter its aspect ratio, final rescale the shape guaranteed to be
    divided by scale_divisor.

    Required Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_polygons (optional)


    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_polygons (optional)

    Added Keys:

    - scale
    - scale_factor
    - keep_ratio

    Args:
        short_size (int): Target shorter size before jittering the aspect
            ratio. Defaults to 736.
        short_size_jitter_range (tuple(float, float)): Range of the ratio used
            to jitter the target shorter size. Defaults to (0.7, 1.3).
        aspect_ratio_jitter_range (tuple(float, float)): Range of the ratio
            used to jitter its aspect ratio. Defaults to (0.9, 1.1).
        scale_divisor (int): The scale divisor. Defaults to 1.
        resize_cfg (dict):  (dict): Config to construct the Resize transform.
            Refer to ``Resize`` for detail. Defaults to
            ``dict(type='Resize')``.
    """

    def __init__(self,
                 short_size: int = 736,
                 ratio_range: Tuple[float, float] = (0.7, 1.3),
                 aspect_ratio_range: Tuple[float, float] = (0.9, 1.1),
                 scale_divisor: int = 1,
                 resize_cfg: Dict = dict(type='Resize')) -> None:
        super().__init__()
        self.short_size = short_size
        self.ratio_range = ratio_range
        self.aspect_ratio_range = aspect_ratio_range
        self.resize_cfg = resize_cfg
        # create a empty Reisize object
        resize_cfg.update(dict(scale=0))
        self.resize = TRANSFORMS.build(resize_cfg)
        self.scale_divisor = scale_divisor

    def _sample_from_range(self, range: Tuple[float, float]) -> float:
        """A ratio will be randomly sampled from the range specified by
        ``range``.

        Args:
            ratio_range (tuple[float]): The minimum and maximum ratio.

        Returns:
            float: A ratio randomly sampled from the range.
        """
        min_value, max_value = min(range), max(range)
        value = np.random.random_sample() * (max_value - min_value) + min_value
        return value

    def transform(self, results: Dict) -> Dict:
        h, w = results['img'].shape[:2]
        ratio = self._sample_from_range(self.ratio_range)
        scale = (ratio * self.short_size) / min(h, w)

        aspect = self._sample_from_range(self.aspect_ratio_range)
        h_scale = scale * math.sqrt(aspect)
        w_scale = scale / math.sqrt(aspect)
        new_h = round(h * h_scale)
        new_w = round(w * w_scale)

        new_h = math.ceil(new_h / self.scale_divisor) * self.scale_divisor
        new_w = math.ceil(new_w / self.scale_divisor) * self.scale_divisor
        self.resize.scale = (new_w, new_h)
        return self.resize(results)

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(short_size = {self.short_size}, '
        repr_str += f'ratio_range = {self.ratio_range}, '
        repr_str += f'aspect_ratio_range = {self.aspect_ratio_range}, '
        repr_str += f'scale_divisor = {self.scale_divisor}, '
        repr_str += f'resize_cfg = {self.resize_cfg})'
        return repr_str

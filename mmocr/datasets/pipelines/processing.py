# Copyright (c) OpenMMLab. All rights reserved.

import math
import random
from typing import Dict, List, Tuple

import cv2
import mmcv
import numpy as np
from mmcv.image.geometric import _scale_size
from mmcv.transforms import Resize as MMCV_Resize
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.utils import cache_randomness
from shapely.geometry import Polygon as plg

import mmocr.core.evaluation.utils as eval_utils
from mmocr.registry import TRANSFORMS
from mmocr.utils import (bbox2poly, crop_polygon, poly2bbox, rescale_bboxes,
                         rescale_polygon)
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
            float: The randomized factor
        """
        return np.random.randint(0, self.factor + 1)

    def transform(self, results: Dict) -> Dict:
        """Applying pyramid rescale on results.

        Args:
            results (dict): Result dict containing the data to transform.

        Returns:
            Dict: The transformed data
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
    ``scale_factor``. Bboxes and polygons are then resized
    with the same scale factor.
    if ``scale`` and ``scale_factor`` are both set, it will use ``scale`` to
    resize.

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
        scale (int or tuple): Image scales for resizing. Defaults to None
        scale_factor (float or tuple[float]): Scale factors for resizing. It's
        either a factor applicable to both dimensions or in the form of
        (scale_w, scale_h). Defaults to None.
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Defaults to False.
        clip_object_border (bool): Whether to clip the objects
            outside the border of the image. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'bilinear'.
    """

    def _resize_bboxes(self, results: dict) -> None:
        """Resize bounding boxes with ``results['scale_factor']``."""
        if results.get('gt_bboxes', None) is not None:
            bboxes = rescale_bboxes(results['gt_bboxes'],
                                    results['scale_factor'])
            if self.clip_object_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0,
                                          results['img_shape'][1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0,
                                          results['img_shape'][0])
            results['gt_bboxes'] = bboxes.astype(np.float32)

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
                    polygons_resize.append(np.zeros_like(polygons[idx]))
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

        if self.scale:
            results['scale'] = self.scale
        else:
            img_shape = results['img'].shape[:2]
            results['scale'] = _scale_size(img_shape[::-1], self.scale_factor)
        self._resize_img(results)
        self._resize_bboxes(results)
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
            results['img_shape'] = (results['img'].shape[0],
                                    results['img'].shape[1])
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
        # TODO Add random_crop_flip_bboxes (will be added after the poly2box
        # and box2poly have been merged)
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
            for polygon in polygons:
                ppi = plg(polygon.reshape(-1, 2))
                # TODO Move this eval_utils to point_utils?
                ppiou = eval_utils.poly_intersection(ppi, pp)
                if np.abs(ppiou - float(ppi.area)) > self.epsilon and \
                        np.abs(ppiou) > self.epsilon:
                    success_flag = False
                    break
                if np.abs(ppiou - float(ppi.area)) < self.epsilon:
                    polys_new.append(polygon)
                else:
                    polys_keep.append(polygon)

            if success_flag:
                break

        cropped = image[ymin:ymax, xmin:xmax, :]
        select_type = self._random_flip_type()
        print(select_type)
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
            results['gt_polygons'] = polygons

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

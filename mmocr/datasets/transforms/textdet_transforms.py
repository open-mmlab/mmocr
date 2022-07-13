# Copyright (c) OpenMMLab. All rights reserved.
import math
import random
from typing import Dict, List, Sequence, Tuple, Union

import cv2
import mmcv
import numpy as np
from mmcv.transforms import RandomFlip as MMCV_RandomFlip
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.utils import avoid_cache_randomness, cache_randomness
from shapely.geometry import Polygon as plg

from mmocr.registry import TRANSFORMS
from mmocr.utils import (bbox2poly, crop_polygon, poly2bbox, poly2shapely,
                         poly_intersection, poly_make_valid, shapely2poly)


@TRANSFORMS.register_module()
@avoid_cache_randomness
class BoundedScaleAspectJitter(BaseTransform):
    """First randomly rescale the image so that the longside and shortside of
    the image are around the bound; then jitter its aspect ratio.

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
        long_size_bound (int): The approximate bound for long size.
        short_size_bound (int): The approximate bound for short size.
        size_jitter_range (tuple(float, float)): Range of the ratio used
            to jitter the size. Defaults to (0.7, 1.3).
        aspect_ratio_jitter_range (tuple(float, float)): Range of the ratio
            used to jitter its aspect ratio. Defaults to (0.9, 1.1).
        resize_type (str): The type of resize class to use. Defaults to
            "Resize".
        **resize_kwargs: Other keyword arguments for the ``resize_type``.
    """

    def __init__(
        self,
        long_size_bound: int,
        short_size_bound: int,
        ratio_range: Tuple[float, float] = (0.7, 1.3),
        aspect_ratio_range: Tuple[float, float] = (0.9, 1.1),
        resize_type: str = 'Resize',
        **resize_kwargs,
    ) -> None:
        super().__init__()
        self.ratio_range = ratio_range
        self.aspect_ratio_range = aspect_ratio_range
        self.long_size_bound = long_size_bound
        self.short_size_bound = short_size_bound
        self.resize_cfg = dict(type=resize_type, **resize_kwargs)
        # create an empty Reisize object
        self.resize_cfg.update(dict(scale=0))
        self.resize = TRANSFORMS.build(self.resize_cfg)

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
        new_scale = 1
        if max(h, w) > self.long_size_bound:
            new_scale = self.long_size_bound / max(h, w)
        jitter_ratio = self._sample_from_range(self.ratio_range)
        jitter_ratio = new_scale * jitter_ratio
        if min(h, w) * jitter_ratio <= self.short_size_bound:
            jitter_ratio = (self.short_size_bound + 10) * 1.0 / min(h, w)
        aspect = self._sample_from_range(self.aspect_ratio_range)
        h_scale = jitter_ratio * math.sqrt(aspect)
        w_scale = jitter_ratio / math.sqrt(aspect)
        new_h = int(h * h_scale)
        new_w = int(w * w_scale)

        self.resize.scale = (new_w, new_h)
        return self.resize(results)

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(long_size_bound = {self.long_size_bound}, '
        repr_str += f'short_size_bound = {self.short_size_bound}, '
        repr_str += f'ratio_range = {self.ratio_range}, '
        repr_str += f'aspect_ratio_range = {self.aspect_ratio_range}, '
        repr_str += f'resize_cfg = {self.resize_cfg})'
        return repr_str


@TRANSFORMS.register_module()
class FixInvalidPolygon(BaseTransform):
    """Fix invalid polygons in the dataset.

    Required Keys:

    - gt_polygons
    - gt_ignored

    Modified Keys:

    - gt_polygons
    - gt_ignored

    Args:
        mode (str): The mode of fixing invalid polygons. Options are 'fix' and
            'ignore'. For the 'fix' mode, the transform will try to fix
            the invalid polygons to a valid one by eliminating the
            self-intersection. For the 'ignore' mode, the invalid polygons
            will be ignored during training. Defaults to 'fix'.
        min_poly_points (int): Minimum number of the coordinate points in a
            polygon. Defaults to 3.
    """

    def __init__(self, mode: str = 'fix', min_poly_points: int = 3) -> None:
        super().__init__()
        self.mode = mode
        self.min_poly_points = min_poly_points
        assert self.mode in [
            'fix', 'ignore'
        ], f"Supported modes are 'fix' and 'ignore', but got {self.mode}"

    def transform(self, results: Dict) -> Dict:
        """Fix invalid polygons.

        Args:
            results (dict): Result dict containing the data to transform.

        Returns:
            dict: The transformed data.
        """
        if results.get('gt_polygons', None) is not None:
            for idx, polygon in enumerate(results['gt_polygons']):
                if not (len(polygon) >= self.min_poly_points * 2
                        and len(polygon) % 2 == 0):
                    results['gt_polygons'][idx] = bbox2poly(
                        results['gt_bboxes'][idx])
                    continue
                polygon = poly2shapely(polygon)
                if not polygon.is_valid:
                    if self.mode == 'fix':
                        polygon = poly_make_valid(polygon)
                        polygon = shapely2poly(polygon)
                        results['gt_polygons'][idx] = polygon
                    elif self.mode == 'ignore':
                        results['gt_ignored'][idx] = True
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(mode = "{self.mode}")'
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
        """Flip images, bounding boxes and polygons.

        Args:
            results (dict): Result dict containing the data to transform.
        """
        super()._flip(results)
        # flip polygons
        if results.get('gt_polygons', None) is not None:
            results['gt_polygons'] = self.flip_polygons(
                results['gt_polygons'], results['img'].shape[:2],
                results['flip_direction'])


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
        resize_type (str): The type of resize class to use. Defaults to
            "Resize".
        **resize_kwargs: Other keyword arguments for the ``resize_type``.
    """

    def __init__(self,
                 short_size: int = 736,
                 ratio_range: Tuple[float, float] = (0.7, 1.3),
                 aspect_ratio_range: Tuple[float, float] = (0.9, 1.1),
                 scale_divisor: int = 1,
                 resize_type: str = 'Resize',
                 **resize_kwargs) -> None:

        super().__init__()
        self.short_size = short_size
        self.ratio_range = ratio_range
        self.aspect_ratio_range = aspect_ratio_range
        self.resize_cfg = dict(type=resize_type, **resize_kwargs)

        # create a empty Reisize object
        self.resize_cfg.update(dict(scale=0))
        self.resize = TRANSFORMS.build(self.resize_cfg)
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
        """Short Scale Aspect Jitter.
        Args:
            results (dict): Result dict containing the data to transform.

        Returns:
            dict: The transformed data.
        """
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
                ppiou = poly_intersection(ppi, pp)
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

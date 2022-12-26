# Copyright (c) OpenMMLab. All rights reserved.
import math
import random
from typing import Dict, List, Optional, Tuple

import cv2
import mmcv
import numpy as np
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.utils import cache_randomness

from mmocr.registry import TRANSFORMS


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
        resize_type (str): The type of resize class to use. Defaults to
            "Resize".
        **resize_kwargs: Other keyword arguments for the ``resize_type``.
    """

    def __init__(self,
                 height: int,
                 min_width: Optional[int] = None,
                 max_width: Optional[int] = None,
                 width_divisor: int = 1,
                 resize_type: str = 'Resize',
                 **resize_kwargs) -> None:

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
        self.resize_cfg = dict(type=resize_type, **resize_kwargs)
        self.resize_cfg.update(dict(scale=0))
        self.resize = TRANSFORMS.build(self.resize_cfg)

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
class TextRecogGeneralAug(BaseTransform):
    """A general geometric augmentation tool for text images in the CVPR 2020
    paper "Learn to Augment: Joint Data Augmentation and Network Optimization
    for Text Recognition". It applies distortion, stretching, and perspective
    transforms to an image.

    This implementation is adapted from
    https://github.com/RubanSeven/Text-Image-Augmentation-python/blob/master/augment.py  # noqa

    Required Keys:

    - img

    Modified Keys:

    - img
    - img_shape
    """  # noqa

    def transform(self, results: Dict) -> Dict:
        """Call function to pad images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        h, w = results['img'].shape[:2]
        if h >= 20 and w >= 20:
            results['img'] = self.tia_distort(results['img'],
                                              random.randint(3, 6))
            results['img'] = self.tia_stretch(results['img'],
                                              random.randint(3, 6))
        h, w = results['img'].shape[:2]
        if h >= 5 and w >= 5:
            results['img'] = self.tia_perspective(results['img'])
        results['img_shape'] = results['img'].shape[:2]
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += '()'
        return repr_str

    def tia_distort(self, img: np.ndarray, segment: int = 4) -> np.ndarray:
        """Image distortion."""
        img_h, img_w = img.shape[:2]

        cut = img_w // segment
        thresh = cut // 3

        src_pts = list()
        dst_pts = list()

        src_pts.append([0, 0])
        src_pts.append([img_w, 0])
        src_pts.append([img_w, img_h])
        src_pts.append([0, img_h])

        dst_pts.append([np.random.randint(thresh), np.random.randint(thresh)])
        dst_pts.append(
            [img_w - np.random.randint(thresh),
             np.random.randint(thresh)])
        dst_pts.append([
            img_w - np.random.randint(thresh),
            img_h - np.random.randint(thresh)
        ])
        dst_pts.append(
            [np.random.randint(thresh), img_h - np.random.randint(thresh)])

        half_thresh = thresh * 0.5

        for cut_idx in np.arange(1, segment, 1):
            src_pts.append([cut * cut_idx, 0])
            src_pts.append([cut * cut_idx, img_h])
            dst_pts.append([
                cut * cut_idx + np.random.randint(thresh) - half_thresh,
                np.random.randint(thresh) - half_thresh
            ])
            dst_pts.append([
                cut * cut_idx + np.random.randint(thresh) - half_thresh,
                img_h + np.random.randint(thresh) - half_thresh
            ])

        dst = self.warp_mls(img, src_pts, dst_pts, img_w, img_h)

        return dst

    def tia_stretch(self, img: np.ndarray, segment: int = 4) -> np.ndarray:
        """Image stretching."""
        img_h, img_w = img.shape[:2]

        cut = img_w // segment
        thresh = cut * 4 // 5

        src_pts = list()
        dst_pts = list()

        src_pts.append([0, 0])
        src_pts.append([img_w, 0])
        src_pts.append([img_w, img_h])
        src_pts.append([0, img_h])

        dst_pts.append([0, 0])
        dst_pts.append([img_w, 0])
        dst_pts.append([img_w, img_h])
        dst_pts.append([0, img_h])

        half_thresh = thresh * 0.5

        for cut_idx in np.arange(1, segment, 1):
            move = np.random.randint(thresh) - half_thresh
            src_pts.append([cut * cut_idx, 0])
            src_pts.append([cut * cut_idx, img_h])
            dst_pts.append([cut * cut_idx + move, 0])
            dst_pts.append([cut * cut_idx + move, img_h])

        dst = self.warp_mls(img, src_pts, dst_pts, img_w, img_h)

        return dst

    def tia_perspective(self, img: np.ndarray) -> np.ndarray:
        """Image perspective transformation."""
        img_h, img_w = img.shape[:2]

        thresh = img_h // 2

        src_pts = list()
        dst_pts = list()

        src_pts.append([0, 0])
        src_pts.append([img_w, 0])
        src_pts.append([img_w, img_h])
        src_pts.append([0, img_h])

        dst_pts.append([0, np.random.randint(thresh)])
        dst_pts.append([img_w, np.random.randint(thresh)])
        dst_pts.append([img_w, img_h - np.random.randint(thresh)])
        dst_pts.append([0, img_h - np.random.randint(thresh)])

        dst = self.warp_mls(img, src_pts, dst_pts, img_w, img_h)

        return dst

    def warp_mls(self,
                 src: np.ndarray,
                 src_pts: List[int],
                 dst_pts: List[int],
                 dst_w: int,
                 dst_h: int,
                 trans_ratio: float = 1.) -> np.ndarray:
        """Warp the image."""
        rdx, rdy = self._calc_delta(dst_w, dst_h, src_pts, dst_pts, 100)
        return self._gen_img(src, rdx, rdy, dst_w, dst_h, 100, trans_ratio)

    def _calc_delta(self, dst_w: int, dst_h: int, src_pts: List[int],
                    dst_pts: List[int],
                    grid_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute delta."""

        pt_count = len(dst_pts)
        rdx = np.zeros((dst_h, dst_w))
        rdy = np.zeros((dst_h, dst_w))
        w = np.zeros(pt_count, dtype=np.float32)

        if pt_count < 2:
            return

        i = 0
        while 1:
            if dst_w <= i < dst_w + grid_size - 1:
                i = dst_w - 1
            elif i >= dst_w:
                break

            j = 0
            while 1:
                if dst_h <= j < dst_h + grid_size - 1:
                    j = dst_h - 1
                elif j >= dst_h:
                    break

                sw = 0
                swp = np.zeros(2, dtype=np.float32)
                swq = np.zeros(2, dtype=np.float32)
                new_pt = np.zeros(2, dtype=np.float32)
                cur_pt = np.array([i, j], dtype=np.float32)

                k = 0
                for k in range(pt_count):
                    if i == dst_pts[k][0] and j == dst_pts[k][1]:
                        break

                    w[k] = 1. / ((i - dst_pts[k][0]) * (i - dst_pts[k][0]) +
                                 (j - dst_pts[k][1]) * (j - dst_pts[k][1]))

                    sw += w[k]
                    swp = swp + w[k] * np.array(dst_pts[k])
                    swq = swq + w[k] * np.array(src_pts[k])

                if k == pt_count - 1:
                    pstar = 1 / sw * swp
                    qstar = 1 / sw * swq

                    miu_s = 0
                    for k in range(pt_count):
                        if i == dst_pts[k][0] and j == dst_pts[k][1]:
                            continue
                        pt_i = dst_pts[k] - pstar
                        miu_s += w[k] * np.sum(pt_i * pt_i)

                    cur_pt -= pstar
                    cur_pt_j = np.array([-cur_pt[1], cur_pt[0]])

                    for k in range(pt_count):
                        if i == dst_pts[k][0] and j == dst_pts[k][1]:
                            continue

                        pt_i = dst_pts[k] - pstar
                        pt_j = np.array([-pt_i[1], pt_i[0]])

                        tmp_pt = np.zeros(2, dtype=np.float32)
                        tmp_pt[0] = (
                            np.sum(pt_i * cur_pt) * src_pts[k][0] -
                            np.sum(pt_j * cur_pt) * src_pts[k][1])
                        tmp_pt[1] = (-np.sum(pt_i * cur_pt_j) * src_pts[k][0] +
                                     np.sum(pt_j * cur_pt_j) * src_pts[k][1])
                        tmp_pt *= (w[k] / miu_s)
                        new_pt += tmp_pt

                    new_pt += qstar
                else:
                    new_pt = src_pts[k]

                rdx[j, i] = new_pt[0] - i
                rdy[j, i] = new_pt[1] - j

                j += grid_size
            i += grid_size
        return rdx, rdy

    def _gen_img(self, src: np.ndarray, rdx: np.ndarray, rdy: np.ndarray,
                 dst_w: int, dst_h: int, grid_size: int,
                 trans_ratio: float) -> np.ndarray:
        """Generate the image based on delta."""

        src_h, src_w = src.shape[:2]
        dst = np.zeros_like(src, dtype=np.float32)

        for i in np.arange(0, dst_h, grid_size):
            for j in np.arange(0, dst_w, grid_size):
                ni = i + grid_size
                nj = j + grid_size
                w = h = grid_size
                if ni >= dst_h:
                    ni = dst_h - 1
                    h = ni - i + 1
                if nj >= dst_w:
                    nj = dst_w - 1
                    w = nj - j + 1

                di = np.reshape(np.arange(h), (-1, 1))
                dj = np.reshape(np.arange(w), (1, -1))
                delta_x = self._bilinear_interp(di / h, dj / w, rdx[i, j],
                                                rdx[i, nj], rdx[ni, j],
                                                rdx[ni, nj])
                delta_y = self._bilinear_interp(di / h, dj / w, rdy[i, j],
                                                rdy[i, nj], rdy[ni, j],
                                                rdy[ni, nj])
                nx = j + dj + delta_x * trans_ratio
                ny = i + di + delta_y * trans_ratio
                nx = np.clip(nx, 0, src_w - 1)
                ny = np.clip(ny, 0, src_h - 1)
                nxi = np.array(np.floor(nx), dtype=np.int32)
                nyi = np.array(np.floor(ny), dtype=np.int32)
                nxi1 = np.array(np.ceil(nx), dtype=np.int32)
                nyi1 = np.array(np.ceil(ny), dtype=np.int32)

                if len(src.shape) == 3:
                    x = np.tile(np.expand_dims(ny - nyi, axis=-1), (1, 1, 3))
                    y = np.tile(np.expand_dims(nx - nxi, axis=-1), (1, 1, 3))
                else:
                    x = ny - nyi
                    y = nx - nxi
                dst[i:i + h,
                    j:j + w] = self._bilinear_interp(x, y, src[nyi, nxi],
                                                     src[nyi, nxi1],
                                                     src[nyi1, nxi], src[nyi1,
                                                                         nxi1])

        dst = np.clip(dst, 0, 255)
        dst = np.array(dst, dtype=np.uint8)

        return dst

    @staticmethod
    def _bilinear_interp(x, y, v11, v12, v21, v22):
        return (v11 * (1 - y) + v12 * y) * (1 - x) + (v21 *
                                                      (1 - y) + v22 * y) * x


@TRANSFORMS.register_module()
class TextRecogRandomCrop(BaseTransform):
    """Randomly crop the image's height, either from top or bottom.

    Adapted from
    https://github.com/PaddlePaddle/PaddleOCR/blob/release%2F2.6/ppocr/data/imaug/rec_img_aug.py  # noqa

    Required Keys:

    - img

    Modified Keys:

    - img
    - img_shape

    Args:
        crop_min (int): Minimum pixel(s) to crop. Defaults to 1.
        crop_max (int): Maximum pixel(s) to crop. Defaults to 8.
    """

    def __init__(
        self,
        min_pixels: int = 1,
        max_pixels: int = 8,
    ) -> None:
        super().__init__()
        assert max_pixels >= min_pixels
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    def transform(self, results: Dict) -> Dict:
        """Transform function to crop images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Cropped results.
        """
        h = results['img'].shape[0]
        top_crop = int(random.randint(self.min_pixels, self.max_pixels))
        top_crop = min(top_crop, h - 1)
        ratio = random.randint(0, 1)
        img = results['img'].copy()
        if ratio:
            img = img[top_crop:h, :, :]
        else:
            img = img[0:h - top_crop, :, :]
        results['img_shape'] = img.shape[:2]
        results['img'] = img
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'( min_pixels = {self.min_pixels}, '
        repr_str += f'max_pixels = {self.max_pixels})'
        return repr_str


@TRANSFORMS.register_module()
class ImageContentJitter(BaseTransform):
    """Jitter the image contents.

    Adapted from
    https://github.com/PaddlePaddle/PaddleOCR/blob/release%2F2.6/ppocr/data/imaug/rec_img_aug.py  # noqa

    Required Keys:

    - img

    Modified Keys:

    - img
    """

    def transform(self, results: Dict) -> Dict:
        """Transform function to jitter images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Jittered results.
        """
        h, w = results['img'].shape[:2]
        img = results['img'].copy()
        if h > 10 and w > 10:
            thres = min(h, w)
            jitter_range = int(random.random() * thres * 0.01)
            for i in range(jitter_range):
                img[i:, i:, :] = img[:h - i, :w - i, :]
        results['img'] = img
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += '()'
        return repr_str


@TRANSFORMS.register_module()
class ReversePixels(BaseTransform):
    """Reverse image pixels.

    Adapted from
    https://github.com/PaddlePaddle/PaddleOCR/blob/release%2F2.6/ppocr/data/imaug/rec_img_aug.py  # noqa

    Required Keys:

    - img

    Modified Keys:

    - img
    """

    def transform(self, results: Dict) -> Dict:
        """Transform function to reverse image pixels.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Reversed results.
        """
        results['img'] = 255 - results['img'].copy()
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += '()'
        return repr_str

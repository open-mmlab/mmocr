# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, Optional, Tuple

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

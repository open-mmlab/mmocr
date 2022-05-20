# Copyright (c) OpenMMLab. All rights reserved.
import math

import mmcv
import numpy as np
from mmdet.core import PolygonMasks
from mmdet.datasets.pipelines.transforms import Resize

from mmocr.registry import TRANSFORMS
from mmocr.utils import check_argument


@TRANSFORMS.register_module()
class ScaleAspectJitter(Resize):
    """Resize image and segmentation mask encoded by coordinates.

    Allowed resize types are `around_min_img_scale`, `long_short_bound`, and
    `indep_sample_in_range`.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=False,
                 resize_type='around_min_img_scale',
                 aspect_ratio_range=None,
                 long_size_bound=None,
                 short_size_bound=None,
                 scale_range=None):
        super().__init__(
            img_scale=img_scale,
            multiscale_mode=multiscale_mode,
            ratio_range=ratio_range,
            keep_ratio=keep_ratio)
        assert not keep_ratio
        assert resize_type in [
            'around_min_img_scale', 'long_short_bound', 'indep_sample_in_range'
        ]
        self.resize_type = resize_type

        if resize_type == 'indep_sample_in_range':
            assert ratio_range is None
            assert aspect_ratio_range is None
            assert short_size_bound is None
            assert long_size_bound is None
            assert scale_range is not None
        else:
            assert scale_range is None
            assert isinstance(ratio_range, tuple)
            assert isinstance(aspect_ratio_range, tuple)
            assert check_argument.equal_len(ratio_range, aspect_ratio_range)

            if resize_type in ['long_short_bound']:
                assert short_size_bound is not None
                assert long_size_bound is not None

        self.aspect_ratio_range = aspect_ratio_range
        self.long_size_bound = long_size_bound
        self.short_size_bound = short_size_bound
        self.scale_range = scale_range

    @staticmethod
    def sample_from_range(range):
        assert len(range) == 2
        min_value, max_value = min(range), max(range)
        value = np.random.random_sample() * (max_value - min_value) + min_value

        return value

    def _random_scale(self, results):

        if self.resize_type == 'indep_sample_in_range':
            w = self.sample_from_range(self.scale_range)
            h = self.sample_from_range(self.scale_range)
            results['scale'] = (int(w), int(h))  # (w,h)
            results['scale_idx'] = None
            return
        h, w = results['img'].shape[0:2]
        if self.resize_type == 'long_short_bound':
            scale1 = 1
            if max(h, w) > self.long_size_bound:
                scale1 = self.long_size_bound / max(h, w)
            scale2 = self.sample_from_range(self.ratio_range)
            scale = scale1 * scale2
            if min(h, w) * scale <= self.short_size_bound:
                scale = (self.short_size_bound + 10) * 1.0 / min(h, w)
        elif self.resize_type == 'around_min_img_scale':
            short_size = min(self.img_scale[0])
            ratio = self.sample_from_range(self.ratio_range)
            scale = (ratio * short_size) / min(h, w)
        else:
            raise NotImplementedError

        aspect = self.sample_from_range(self.aspect_ratio_range)
        h_scale = scale * math.sqrt(aspect)
        w_scale = scale / math.sqrt(aspect)
        results['scale'] = (int(w * w_scale), int(h * h_scale))  # (w,h)
        results['scale_idx'] = None


@TRANSFORMS.register_module()
class SquareResizePad:

    def __init__(self,
                 target_size,
                 pad_ratio=0.6,
                 pad_with_fixed_color=False,
                 pad_value=(0, 0, 0)):
        """Resize or pad images to be square shape.

        Args:
            target_size (int): The target size of square shaped image.
            pad_with_fixed_color (bool): The flag for whether to pad rotated
               image with fixed value. If set to False, the rescales image will
               be padded onto cropped image.
            pad_value (tuple(int)): The color value for padding rotated image.
        """
        assert isinstance(target_size, int)
        assert isinstance(pad_ratio, float)
        assert isinstance(pad_with_fixed_color, bool)
        assert isinstance(pad_value, tuple)

        self.target_size = target_size
        self.pad_ratio = pad_ratio
        self.pad_with_fixed_color = pad_with_fixed_color
        self.pad_value = pad_value

    def resize_img(self, img, keep_ratio=True):
        h, w, _ = img.shape
        if keep_ratio:
            t_h = self.target_size if h >= w else int(h * self.target_size / w)
            t_w = self.target_size if h <= w else int(w * self.target_size / h)
        else:
            t_h = t_w = self.target_size
        img = mmcv.imresize(img, (t_w, t_h))
        return img, (t_h, t_w)

    def square_pad(self, img):
        h, w = img.shape[:2]
        if h == w:
            return img, (0, 0)
        pad_size = max(h, w)
        if self.pad_with_fixed_color:
            expand_img = np.ones((pad_size, pad_size, 3), dtype=np.uint8)
            expand_img[:] = self.pad_value
        else:
            (h_ind, w_ind) = (np.random.randint(0, h * 7 // 8),
                              np.random.randint(0, w * 7 // 8))
            img_cut = img[h_ind:(h_ind + h // 9), w_ind:(w_ind + w // 9)]
            expand_img = mmcv.imresize(img_cut, (pad_size, pad_size))
        if h > w:
            y0, x0 = 0, (h - w) // 2
        else:
            y0, x0 = (w - h) // 2, 0
        expand_img[y0:y0 + h, x0:x0 + w] = img
        offset = (x0, y0)

        return expand_img, offset

    def square_pad_mask(self, points, offset):
        x0, y0 = offset
        pad_points = points.copy()
        pad_points[::2] = pad_points[::2] + x0
        pad_points[1::2] = pad_points[1::2] + y0
        return pad_points

    def __call__(self, results):
        img = results['img']

        if np.random.random_sample() < self.pad_ratio:
            img, out_size = self.resize_img(img, keep_ratio=True)
            img, offset = self.square_pad(img)
        else:
            img, out_size = self.resize_img(img, keep_ratio=False)
            offset = (0, 0)

        results['img'] = img
        results['img_shape'] = img.shape

        for key in results.get('mask_fields', []):
            if len(results[key].masks) == 0:
                continue
            results[key] = results[key].resize(out_size)
            masks = results[key].masks
            processed_masks = []
            for mask in masks:
                square_pad_mask = self.square_pad_mask(mask[0], offset)
                processed_masks.append([square_pad_mask])

            results[key] = PolygonMasks(processed_masks, *(img.shape[:2]))

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

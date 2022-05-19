# Copyright (c) OpenMMLab. All rights reserved.
import math

import mmcv
import numpy as np
from mmdet.core import BitmapMasks, PolygonMasks
from mmdet.datasets.pipelines.transforms import Resize

import mmocr.core.evaluation.utils as eval_utils
from mmocr.registry import TRANSFORMS
from mmocr.utils import check_argument


@TRANSFORMS.register_module()
class RandomCropInstances:
    """Randomly crop images and make sure to contain text instances.

    Args:
        target_size (tuple or int): (height, width)
        positive_sample_ratio (float): The probability of sampling regions
            that go through positive regions.
    """

    def __init__(
            self,
            target_size,
            instance_key,
            mask_type='inx0',  # 'inx0' or 'union_all'
            positive_sample_ratio=5.0 / 8.0):

        assert mask_type in ['inx0', 'union_all']

        self.mask_type = mask_type
        self.instance_key = instance_key
        self.positive_sample_ratio = positive_sample_ratio
        self.target_size = target_size if (target_size is None or isinstance(
            target_size, tuple)) else (target_size, target_size)

    def sample_offset(self, img_gt, img_size):
        h, w = img_size
        t_h, t_w = self.target_size

        # target size is bigger than origin size
        t_h = t_h if t_h < h else h
        t_w = t_w if t_w < w else w
        if (img_gt is not None
                and np.random.random_sample() < self.positive_sample_ratio
                and np.max(img_gt) > 0):

            # make sure to crop the positive region

            # the minimum top left to crop positive region (h,w)
            tl = np.min(np.where(img_gt > 0), axis=1) - (t_h, t_w)
            tl[tl < 0] = 0
            # the maximum top left to crop positive region
            br = np.max(np.where(img_gt > 0), axis=1) - (t_h, t_w)
            br[br < 0] = 0
            # if br is too big so that crop the outside region of img
            br[0] = min(br[0], h - t_h)
            br[1] = min(br[1], w - t_w)
            #
            h = np.random.randint(tl[0], br[0]) if tl[0] < br[0] else 0
            w = np.random.randint(tl[1], br[1]) if tl[1] < br[1] else 0
        else:
            # make sure not to crop outside of img

            h = np.random.randint(0, h - t_h) if h - t_h > 0 else 0
            w = np.random.randint(0, w - t_w) if w - t_w > 0 else 0

        return (h, w)

    @staticmethod
    def crop_img(img, offset, target_size):
        h, w = img.shape[:2]
        br = np.min(
            np.stack((np.array(offset) + np.array(target_size), np.array(
                (h, w)))),
            axis=0)
        return img[offset[0]:br[0], offset[1]:br[1]], np.array(
            [offset[1], offset[0], br[1], br[0]])

    def crop_bboxes(self, bboxes, canvas_bbox):
        kept_bboxes = []
        kept_inx = []
        canvas_poly = eval_utils.box2polygon(canvas_bbox)
        tl = canvas_bbox[0:2]

        for idx, bbox in enumerate(bboxes):
            poly = eval_utils.box2polygon(bbox)
            area, inters = eval_utils.poly_intersection(
                poly, canvas_poly, return_poly=True)
            if area == 0:
                continue
            xmin, ymin, xmax, ymax = inters.bounds
            kept_bboxes += [
                np.array(
                    [xmin - tl[0], ymin - tl[1], xmax - tl[0], ymax - tl[1]],
                    dtype=np.float32)
            ]
            kept_inx += [idx]

        if len(kept_inx) == 0:
            return np.array([]).astype(np.float32).reshape(0, 4), kept_inx

        return np.stack(kept_bboxes), kept_inx

    @staticmethod
    def generate_mask(gt_mask, type):

        if type == 'inx0':
            return gt_mask.masks[0]
        if type == 'union_all':
            mask = gt_mask.masks[0].copy()
            for idx in range(1, len(gt_mask.masks)):
                mask = np.logical_or(mask, gt_mask.masks[idx])
            return mask

        raise NotImplementedError

    def __call__(self, results):

        gt_mask = results[self.instance_key]
        mask = None
        if len(gt_mask.masks) > 0:
            mask = self.generate_mask(gt_mask, self.mask_type)
        results['crop_offset'] = self.sample_offset(mask,
                                                    results['img'].shape[:2])

        # crop img. bbox = [x1,y1,x2,y2]
        img, bbox = self.crop_img(results['img'], results['crop_offset'],
                                  self.target_size)
        results['img'] = img
        img_shape = img.shape
        results['img_shape'] = img_shape

        # crop masks
        for key in results.get('mask_fields', []):
            results[key] = results[key].crop(bbox)

        # for mask rcnn
        for key in results.get('bbox_fields', []):
            results[key], kept_inx = self.crop_bboxes(results[key], bbox)
            if key == 'gt_bboxes':
                # ignore gt_labels accordingly
                if 'gt_labels' in results:
                    ori_labels = results['gt_labels']
                    ori_inst_num = len(ori_labels)
                    results['gt_labels'] = [
                        ori_labels[idx] for idx in range(ori_inst_num)
                        if idx in kept_inx
                    ]
                # ignore g_masks accordingly
                if 'gt_masks' in results:
                    ori_mask = results['gt_masks'].masks
                    kept_mask = [
                        ori_mask[idx] for idx in range(ori_inst_num)
                        if idx in kept_inx
                    ]
                    target_h, target_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                    if len(kept_inx) > 0:
                        kept_mask = np.stack(kept_mask)
                    else:
                        kept_mask = np.empty((0, target_h, target_w),
                                             dtype=np.float32)
                    results['gt_masks'] = BitmapMasks(kept_mask, target_h,
                                                      target_w)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


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
class RandomCropPolyInstances:
    """Randomly crop images and make sure to contain at least one intact
    instance."""

    def __init__(self,
                 instance_key='gt_masks',
                 crop_ratio=5.0 / 8.0,
                 min_side_ratio=0.4):
        super().__init__()
        self.instance_key = instance_key
        self.crop_ratio = crop_ratio
        self.min_side_ratio = min_side_ratio

    def sample_valid_start_end(self, valid_array, min_len, max_start, min_end):

        assert isinstance(min_len, int)
        assert len(valid_array) > min_len

        start_array = valid_array.copy()
        max_start = min(len(start_array) - min_len, max_start)
        start_array[max_start:] = 0
        start_array[0] = 1
        diff_array = np.hstack([0, start_array]) - np.hstack([start_array, 0])
        region_starts = np.where(diff_array < 0)[0]
        region_ends = np.where(diff_array > 0)[0]
        region_ind = np.random.randint(0, len(region_starts))
        start = np.random.randint(region_starts[region_ind],
                                  region_ends[region_ind])

        end_array = valid_array.copy()
        min_end = max(start + min_len, min_end)
        end_array[:min_end] = 0
        end_array[-1] = 1
        diff_array = np.hstack([0, end_array]) - np.hstack([end_array, 0])
        region_starts = np.where(diff_array < 0)[0]
        region_ends = np.where(diff_array > 0)[0]
        region_ind = np.random.randint(0, len(region_starts))
        end = np.random.randint(region_starts[region_ind],
                                region_ends[region_ind])
        return start, end

    def sample_crop_box(self, img_size, results):
        """Generate crop box and make sure not to crop the polygon instances.

        Args:
            img_size (tuple(int)): The image size (h, w).
            results (dict): The results dict.
        """

        assert isinstance(img_size, tuple)
        h, w = img_size[:2]

        key_masks = results[self.instance_key].masks
        x_valid_array = np.ones(w, dtype=np.int32)
        y_valid_array = np.ones(h, dtype=np.int32)

        selected_mask = key_masks[np.random.randint(0, len(key_masks))]
        selected_mask = selected_mask[0].reshape((-1, 2)).astype(np.int32)
        max_x_start = max(np.min(selected_mask[:, 0]) - 2, 0)
        min_x_end = min(np.max(selected_mask[:, 0]) + 3, w - 1)
        max_y_start = max(np.min(selected_mask[:, 1]) - 2, 0)
        min_y_end = min(np.max(selected_mask[:, 1]) + 3, h - 1)

        for key in results.get('mask_fields', []):
            if len(results[key].masks) == 0:
                continue
            masks = results[key].masks
            for mask in masks:
                assert len(mask) == 1
                mask = mask[0].reshape((-1, 2)).astype(np.int32)
                clip_x = np.clip(mask[:, 0], 0, w - 1)
                clip_y = np.clip(mask[:, 1], 0, h - 1)
                min_x, max_x = np.min(clip_x), np.max(clip_x)
                min_y, max_y = np.min(clip_y), np.max(clip_y)

                x_valid_array[min_x - 2:max_x + 3] = 0
                y_valid_array[min_y - 2:max_y + 3] = 0

        min_w = int(w * self.min_side_ratio)
        min_h = int(h * self.min_side_ratio)

        x1, x2 = self.sample_valid_start_end(x_valid_array, min_w, max_x_start,
                                             min_x_end)
        y1, y2 = self.sample_valid_start_end(y_valid_array, min_h, max_y_start,
                                             min_y_end)

        return np.array([x1, y1, x2, y2])

    def crop_img(self, img, bbox):
        assert img.ndim == 3
        h, w, _ = img.shape
        assert 0 <= bbox[1] < bbox[3] <= h
        assert 0 <= bbox[0] < bbox[2] <= w
        return img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    def __call__(self, results):
        if len(results[self.instance_key].masks) < 1:
            return results
        if np.random.random_sample() < self.crop_ratio:
            crop_box = self.sample_crop_box(results['img'].shape, results)
            results['crop_region'] = crop_box
            img = self.crop_img(results['img'], crop_box)
            results['img'] = img
            results['img_shape'] = img.shape

            # crop and filter masks
            x1, y1, x2, y2 = crop_box
            w = max(x2 - x1, 1)
            h = max(y2 - y1, 1)
            labels = results['gt_labels']
            valid_labels = []
            for key in results.get('mask_fields', []):
                if len(results[key].masks) == 0:
                    continue
                results[key] = results[key].crop(crop_box)
                # filter out polygons beyond crop box.
                masks = results[key].masks
                valid_masks_list = []

                for ind, mask in enumerate(masks):
                    assert len(mask) == 1
                    polygon = mask[0].reshape((-1, 2))
                    if (polygon[:, 0] >
                            -4).all() and (polygon[:, 0] < w + 4).all() and (
                                polygon[:, 1] > -4).all() and (polygon[:, 1] <
                                                               h + 4).all():
                        mask[0][::2] = np.clip(mask[0][::2], 0, w)
                        mask[0][1::2] = np.clip(mask[0][1::2], 0, h)
                        if key == self.instance_key:
                            valid_labels.append(labels[ind])
                        valid_masks_list.append(mask)

                results[key] = PolygonMasks(valid_masks_list, h, w)
            results['gt_labels'] = np.array(valid_labels)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


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


@TRANSFORMS.register_module()
class RandomScaling:

    def __init__(self, size=800, scale=(3. / 4, 5. / 2)):
        """Random scale the image while keeping aspect.

        Args:
            size (int) : Base size before scaling.
            scale (tuple(float)) : The range of scaling.
        """
        assert isinstance(size, int)
        assert isinstance(scale, float) or isinstance(scale, tuple)
        self.size = size
        self.scale = scale if isinstance(scale, tuple) \
            else (1 - scale, 1 + scale)

    def __call__(self, results):
        image = results['img']
        h, w, _ = results['img_shape']

        aspect_ratio = np.random.uniform(min(self.scale), max(self.scale))
        scales = self.size * 1.0 / max(h, w) * aspect_ratio
        scales = np.array([scales, scales])
        out_size = (int(h * scales[1]), int(w * scales[0]))
        image = mmcv.imresize(image, out_size[::-1])

        results['img'] = image
        results['img_shape'] = image.shape

        for key in results.get('mask_fields', []):
            if len(results[key].masks) == 0:
                continue
            results[key] = results[key].resize(out_size)

        return results

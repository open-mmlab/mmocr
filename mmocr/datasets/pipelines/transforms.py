import math

import cv2
import numpy as np
import Polygon as plg
import torchvision.transforms as transforms
from PIL import Image

import mmocr.core.evaluation.utils as eval_utils
from mmdet.core import BitmapMasks, PolygonMasks
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.transforms import Resize
from mmocr.utils import check_argument


@PIPELINES.register_module()
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

        for inx, bbox in enumerate(bboxes):
            poly = eval_utils.box2polygon(bbox)
            area, inters = eval_utils.poly_intersection(poly, canvas_poly)
            if area == 0:
                continue
            xmin, xmax, ymin, ymax = inters.boundingBox()
            kept_bboxes += [
                np.array(
                    [xmin - tl[0], ymin - tl[1], xmax - tl[0], ymax - tl[1]],
                    dtype=np.float32)
            ]
            kept_inx += [inx]

        if len(kept_inx) == 0:
            return np.array([]).astype(np.float32).reshape(0, 4), kept_inx

        return np.stack(kept_bboxes), kept_inx

    @staticmethod
    def generate_mask(gt_mask, type):

        if type == 'inx0':
            return gt_mask.masks[0]
        if type == 'union_all':
            mask = gt_mask.masks[0].copy()
            for inx in range(1, len(gt_mask.masks)):
                mask = np.logical_or(mask, gt_mask.masks[inx])
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
                        ori_labels[inx] for inx in range(ori_inst_num)
                        if inx in kept_inx
                    ]
                # ignore g_masks accordingly
                if 'gt_masks' in results:
                    ori_mask = results['gt_masks'].masks
                    kept_mask = [
                        ori_mask[inx] for inx in range(ori_inst_num)
                        if inx in kept_inx
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


@PIPELINES.register_module()
class RandomRotateTextDet:
    """Randomly rotate images."""

    def __init__(self, rotate_ratio=1.0, max_angle=10):
        self.rotate_ratio = rotate_ratio
        self.max_angle = max_angle

    @staticmethod
    def sample_angle(max_angle):
        angle = np.random.random_sample() * 2 * max_angle - max_angle
        return angle

    @staticmethod
    def rotate_img(img, angle):
        h, w = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        img_target = cv2.warpAffine(
            img, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
        assert img_target.shape == img.shape
        return img_target

    def __call__(self, results):
        if np.random.random_sample() < self.rotate_ratio:
            # rotate imgs
            results['rotated_angle'] = self.sample_angle(self.max_angle)
            img = self.rotate_img(results['img'], results['rotated_angle'])
            results['img'] = img
            img_shape = img.shape
            results['img_shape'] = img_shape

            # rotate masks
            for key in results.get('mask_fields', []):
                masks = results[key].masks
                mask_list = []
                for m in masks:
                    rotated_m = self.rotate_img(m, results['rotated_angle'])
                    mask_list.append(rotated_m)
                results[key] = BitmapMasks(mask_list, *(img_shape[:2]))

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class ColorJitter:
    """An interface for torch color jitter so that it can be invoked in
    mmdetection pipeline."""

    def __init__(self, **kwargs):
        self.transform = transforms.ColorJitter(**kwargs)

    def __call__(self, results):
        # img is bgr
        img = results['img'][..., ::-1]
        img = Image.fromarray(img)
        img = self.transform(img)
        img = np.asarray(img)
        img = img[..., ::-1]
        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
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


@PIPELINES.register_module()
class AffineJitter:
    """An interface for torchvision random affine so that it can be invoked in
    mmdet pipeline."""

    def __init__(self,
                 degrees=4,
                 translate=(0.02, 0.04),
                 scale=(0.9, 1.1),
                 shear=None,
                 resample=False,
                 fillcolor=0):
        self.transform = transforms.RandomAffine(
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            resample=resample,
            fillcolor=fillcolor)

    def __call__(self, results):
        # img is bgr
        img = results['img'][..., ::-1]
        img = Image.fromarray(img)
        img = self.transform(img)
        img = np.asarray(img)
        img = img[..., ::-1]
        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
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


@PIPELINES.register_module()
class RandomRotatePolyInstances:

    def __init__(self,
                 rotate_ratio=0.5,
                 max_angle=10,
                 pad_with_fixed_color=False,
                 pad_value=(0, 0, 0)):
        """Randomly rotate images and polygon masks.

        Args:
            rotate_ratio (float): The ratio of samples to operate rotation.
            max_angle (int): The maximum rotation angle.
            pad_with_fixed_color (bool): The flag for whether to pad rotated
               image with fixed value. If set to False, the rotated image will
               be padded onto cropped image.
            pad_value (tuple(int)): The color value for padding rotated image.
        """
        self.rotate_ratio = rotate_ratio
        self.max_angle = max_angle
        self.pad_with_fixed_color = pad_with_fixed_color
        self.pad_value = pad_value

    def rotate(self, center, points, theta, center_shift=(0, 0)):
        # rotate points.
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

    def cal_canvas_size(self, ori_size, degree):
        assert isinstance(ori_size, tuple)
        angle = degree * math.pi / 180.0
        h, w = ori_size[:2]

        cos = math.cos(angle)
        sin = math.sin(angle)
        canvas_h = int(w * math.fabs(sin) + h * math.fabs(cos))
        canvas_w = int(w * math.fabs(cos) + h * math.fabs(sin))

        canvas_size = (canvas_h, canvas_w)
        return canvas_size

    def sample_angle(self, max_angle):
        angle = np.random.random_sample() * 2 * max_angle - max_angle
        return angle

    def rotate_img(self, img, angle, canvas_size):
        h, w = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        rotation_matrix[0, 2] += int((canvas_size[1] - w) / 2)
        rotation_matrix[1, 2] += int((canvas_size[0] - h) / 2)

        if self.pad_with_fixed_color:
            target_img = cv2.warpAffine(
                img,
                rotation_matrix, (canvas_size[1], canvas_size[0]),
                flags=cv2.INTER_NEAREST,
                borderValue=self.pad_value)
        else:
            mask = np.zeros_like(img)
            (h_ind, w_ind) = (np.random.randint(0, h * 7 // 8),
                              np.random.randint(0, w * 7 // 8))
            img_cut = img[h_ind:(h_ind + h // 9), w_ind:(w_ind + w // 9)]
            img_cut = cv2.resize(img_cut, (canvas_size[1], canvas_size[0]))
            mask = cv2.warpAffine(
                mask,
                rotation_matrix, (canvas_size[1], canvas_size[0]),
                borderValue=[1, 1, 1])
            target_img = cv2.warpAffine(
                img,
                rotation_matrix, (canvas_size[1], canvas_size[0]),
                borderValue=[0, 0, 0])
            target_img = target_img + img_cut * mask

        return target_img

    def __call__(self, results):
        if np.random.random_sample() < self.rotate_ratio:
            img = results['img']
            h, w = img.shape[:2]
            angle = self.sample_angle(self.max_angle)
            canvas_size = self.cal_canvas_size((h, w), angle)
            center_shift = (int(
                (canvas_size[1] - w) / 2), int((canvas_size[0] - h) / 2))

            # rotate image
            results['rotated_poly_angle'] = angle
            img = self.rotate_img(img, angle, canvas_size)
            results['img'] = img
            img_shape = img.shape
            results['img_shape'] = img_shape

            # rotate polygons
            for key in results.get('mask_fields', []):
                if len(results[key].masks) == 0:
                    continue
                masks = results[key].masks
                rotated_masks = []
                for mask in masks:
                    rotated_mask = self.rotate((w / 2, h / 2), mask[0], angle,
                                               center_shift)
                    rotated_masks.append([rotated_mask])

                results[key] = PolygonMasks(rotated_masks, *(img_shape[:2]))

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
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
        img = cv2.resize(img, (t_w, t_h))
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
            expand_img = cv2.resize(img_cut, (pad_size, pad_size))
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


@PIPELINES.register_module()
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
        image = cv2.resize(image, out_size[::-1])

        results['img'] = image
        results['img_shape'] = image.shape

        for key in results.get('mask_fields', []):
            if len(results[key].masks) == 0:
                continue
            results[key] = results[key].resize(out_size)

        return results


@PIPELINES.register_module()
class RandomCropFlip:

    def __init__(self,
                 pad_ratio=0.1,
                 crop_ratio=0.5,
                 iter_num=1,
                 min_area_ratio=0.2):
        """Random crop and flip a patch of the image.

        Args:
            crop_ratio (float): The ratio of cropping.
            iter_num (int): Number of operations.
            min_area_ratio (float): Minimal area ratio between cropped patch
                and original image.
        """
        assert isinstance(crop_ratio, float)
        assert isinstance(iter_num, int)
        assert isinstance(min_area_ratio, float)

        self.pad_ratio = pad_ratio
        self.epsilon = 1e-2
        self.crop_ratio = crop_ratio
        self.iter_num = iter_num
        self.min_area_ratio = min_area_ratio

    def __call__(self, results):
        for i in range(self.iter_num):
            results = self.random_crop_flip(results)
        return results

    def random_crop_flip(self, results):
        image = results['img']
        polygons = results['gt_masks'].masks
        ignore_polygons = results['gt_masks_ignore'].masks
        all_polygons = polygons + ignore_polygons
        if len(polygons) == 0:
            return results

        if np.random.random() >= self.crop_ratio:
            return results

        h, w, _ = results['img_shape']
        area = h * w
        pad_h = int(h * self.pad_ratio)
        pad_w = int(w * self.pad_ratio)
        h_axis, w_axis = self.generate_crop_target(image, all_polygons, pad_h,
                                                   pad_w)
        if len(h_axis) == 0 or len(w_axis) == 0:
            return results

        attempt = 0
        while attempt < 10:
            attempt += 1
            polys_keep = []
            polys_new = []
            ign_polys_keep = []
            ign_polys_new = []
            xx = np.random.choice(w_axis, size=2)
            xmin = np.min(xx) - pad_w
            xmax = np.max(xx) - pad_w
            xmin = np.clip(xmin, 0, w - 1)
            xmax = np.clip(xmax, 0, w - 1)
            yy = np.random.choice(h_axis, size=2)
            ymin = np.min(yy) - pad_h
            ymax = np.max(yy) - pad_h
            ymin = np.clip(ymin, 0, h - 1)
            ymax = np.clip(ymax, 0, h - 1)
            if (xmax - xmin) * (ymax - ymin) < area * self.min_area_ratio:
                # area too small
                continue

            pts = np.stack([[xmin, xmax, xmax, xmin],
                            [ymin, ymin, ymax, ymax]]).T.astype(np.int32)
            pp = plg.Polygon(pts)
            fail_flag = False
            for polygon in polygons:
                ppi = plg.Polygon(polygon[0].reshape(-1, 2))
                ppiou, _ = eval_utils.poly_intersection(ppi, pp)
                if np.abs(ppiou - float(ppi.area())) > self.epsilon and \
                        np.abs(ppiou) > self.epsilon:
                    fail_flag = True
                    break
                elif np.abs(ppiou - float(ppi.area())) < self.epsilon:
                    polys_new.append(polygon)
                else:
                    polys_keep.append(polygon)

            for polygon in ignore_polygons:
                ppi = plg.Polygon(polygon[0].reshape(-1, 2))
                ppiou, _ = eval_utils.poly_intersection(ppi, pp)
                if np.abs(ppiou - float(ppi.area())) > self.epsilon and \
                        np.abs(ppiou) > self.epsilon:
                    fail_flag = True
                    break
                elif np.abs(ppiou - float(ppi.area())) < self.epsilon:
                    ign_polys_new.append(polygon)
                else:
                    ign_polys_keep.append(polygon)

            if fail_flag:
                continue
            else:
                break

        cropped = image[ymin:ymax, xmin:xmax, :]
        select_type = np.random.randint(3)
        if select_type == 0:
            img = np.ascontiguousarray(cropped[:, ::-1])
        elif select_type == 1:
            img = np.ascontiguousarray(cropped[::-1, :])
        else:
            img = np.ascontiguousarray(cropped[::-1, ::-1])
        image[ymin:ymax, xmin:xmax, :] = img
        results['img'] = image

        if len(polys_new) + len(ign_polys_new) != 0:
            height, width, _ = cropped.shape
            if select_type == 0:
                for idx, polygon in enumerate(polys_new):
                    poly = polygon[0].reshape(-1, 2)
                    poly[:, 0] = width - poly[:, 0] + 2 * xmin
                    polys_new[idx] = [poly.reshape(-1, )]
                for idx, polygon in enumerate(ign_polys_new):
                    poly = polygon[0].reshape(-1, 2)
                    poly[:, 0] = width - poly[:, 0] + 2 * xmin
                    ign_polys_new[idx] = [poly.reshape(-1, )]
            elif select_type == 1:
                for idx, polygon in enumerate(polys_new):
                    poly = polygon[0].reshape(-1, 2)
                    poly[:, 1] = height - poly[:, 1] + 2 * ymin
                    polys_new[idx] = [poly.reshape(-1, )]
                for idx, polygon in enumerate(ign_polys_new):
                    poly = polygon[0].reshape(-1, 2)
                    poly[:, 1] = height - poly[:, 1] + 2 * ymin
                    ign_polys_new[idx] = [poly.reshape(-1, )]
            else:
                for idx, polygon in enumerate(polys_new):
                    poly = polygon[0].reshape(-1, 2)
                    poly[:, 0] = width - poly[:, 0] + 2 * xmin
                    poly[:, 1] = height - poly[:, 1] + 2 * ymin
                    polys_new[idx] = [poly.reshape(-1, )]
                for idx, polygon in enumerate(ign_polys_new):
                    poly = polygon[0].reshape(-1, 2)
                    poly[:, 0] = width - poly[:, 0] + 2 * xmin
                    poly[:, 1] = height - poly[:, 1] + 2 * ymin
                    ign_polys_new[idx] = [poly.reshape(-1, )]
            polygons = polys_keep + polys_new
            ignore_polygons = ign_polys_keep + ign_polys_new
            results['gt_masks'] = PolygonMasks(polygons, *(image.shape[:2]))
            results['gt_masks_ignore'] = PolygonMasks(ignore_polygons,
                                                      *(image.shape[:2]))

        return results

    def generate_crop_target(self, image, all_polys, pad_h, pad_w):
        """Generate crop target and make sure not to crop the polygon
        instances.

        Args:
            image (ndarray): The image waited to be crop.
            all_polys (list[list[ndarray]]): All polygons including ground
                truth polygons and ground truth ignored polygons.
            pad_h (int): Padding length of height.
            pad_w (int): Padding length of width.
        Returns:
            h_axis (ndarray): Vertical cropping range.
            w_axis (ndarray): Horizontal cropping range.
        """
        h, w, _ = image.shape
        h_array = np.zeros((h + pad_h * 2), dtype=np.int32)
        w_array = np.zeros((w + pad_w * 2), dtype=np.int32)

        text_polys = []
        for polygon in all_polys:
            rect = cv2.minAreaRect(polygon[0].astype(np.int32).reshape(-1, 2))
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            text_polys.append([box[0], box[1], box[2], box[3]])

        polys = np.array(text_polys, dtype=np.int32)
        for poly in polys:
            poly = np.round(poly, decimals=0).astype(np.int32)
            minx = np.min(poly[:, 0])
            maxx = np.max(poly[:, 0])
            w_array[minx + pad_w:maxx + pad_w] = 1
            miny = np.min(poly[:, 1])
            maxy = np.max(poly[:, 1])
            h_array[miny + pad_h:maxy + pad_h] = 1

        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]
        return h_axis, w_axis

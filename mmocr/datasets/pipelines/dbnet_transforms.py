# Copyright (c) OpenMMLab. All rights reserved.
import imgaug
import imgaug.augmenters as iaa
import mmcv
import numpy as np
from mmdet.core.mask import PolygonMasks
from mmdet.datasets.builder import PIPELINES


class AugmenterBuilder:
    """Build imgaug object according ImgAug argmentations."""

    def __init__(self):
        pass

    def build(self, args, root=True):
        if args is None:
            return None
        if isinstance(args, (int, float, str)):
            return args
        if isinstance(args, list):
            if root:
                sequence = [self.build(value, root=False) for value in args]
                return iaa.Sequential(sequence)
            arg_list = [self.to_tuple_if_list(a) for a in args[1:]]
            return getattr(iaa, args[0])(*arg_list)
        if isinstance(args, dict):
            if 'cls' in args:
                cls = getattr(iaa, args['cls'])
                return cls(
                    **{
                        k: self.to_tuple_if_list(v)
                        for k, v in args.items() if not k == 'cls'
                    })
            else:
                return {
                    key: self.build(value, root=False)
                    for key, value in args.items()
                }
        raise RuntimeError('unknown augmenter arg: ' + str(args))

    def to_tuple_if_list(self, obj):
        if isinstance(obj, list):
            return tuple(obj)
        return obj


@PIPELINES.register_module()
class ImgAug:
    """A wrapper to use imgaug https://github.com/aleju/imgaug.

    Args:
        args ([list[list|dict]]): The argumentation list. For details, please
            refer to imgaug document. Take args=[['Fliplr', 0.5],
            dict(cls='Affine', rotate=[-10, 10]), ['Resize', [0.5, 3.0]]] as an
            example. The args horizontally flip images with probability 0.5,
            followed by random rotation with angles in range [-10, 10], and
            resize with an independent scale in range [0.5, 3.0] for each
            side of images.
        clip_invalid_polys (bool): Whether to clip invalid polygons after
            transformation. False persists to the behavior in DBNet.
    """

    def __init__(self, args=None, clip_invalid_ploys=False):
        self.augmenter_args = args
        self.augmenter = AugmenterBuilder().build(self.augmenter_args)
        self.clip_invalid_polys = clip_invalid_ploys

    def __call__(self, results):
        # img is bgr
        image = results['img']
        aug = None
        shape = image.shape

        if self.augmenter:
            aug = self.augmenter.to_deterministic()
            results['img'] = aug.augment_image(image)
            results['img_shape'] = results['img'].shape
            results['flip'] = 'unknown'  # it's unknown
            results['flip_direction'] = 'unknown'  # it's unknown
            target_shape = results['img_shape']

            self.may_augment_annotation(aug, shape, target_shape, results)

        return results

    def may_augment_annotation(self, aug, shape, target_shape, results):
        if aug is None:
            return results

        # augment polygon mask
        for key in results['mask_fields']:
            if self.clip_invalid_polys:
                masks = self.may_augment_poly(aug, shape, results[key])
                results[key] = PolygonMasks(masks, *target_shape[:2])
            else:
                masks = self.may_augment_poly_legacy(aug, shape, results[key])
                if len(masks) > 0:
                    results[key] = PolygonMasks(masks, *target_shape[:2])

        # augment bbox
        for key in results['bbox_fields']:
            bboxes = self.may_augment_bbox(aug, shape, results[key])
            results[key] = np.zeros(0)
            if len(bboxes) > 0:
                results[key] = np.stack(bboxes)

        return results

    def may_augment_bbox(self, aug, ori_shape, bboxes):
        imgaug_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            imgaug_bboxes.append(
                imgaug.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))
        imgaug_bboxes = aug.augment_bounding_boxes([
            imgaug.BoundingBoxesOnImage(imgaug_bboxes, shape=ori_shape)
        ])[0].clip_out_of_image()

        new_bboxes = []
        for box in imgaug_bboxes.bounding_boxes:
            new_bboxes.append(
                np.array([box.x1, box.y1, box.x2, box.y2], dtype=np.float32))

        return new_bboxes

    def may_augment_poly(self, aug, img_shape, polys):
        imgaug_polys = []
        for poly in polys:
            poly = poly[0]
            poly = poly.reshape(-1, 2)
            imgaug_polys.append(imgaug.Polygon(poly))
        imgaug_polys = aug.augment_polygons(
            [imgaug.PolygonsOnImage(imgaug_polys,
                                    shape=img_shape)])[0].clip_out_of_image()

        new_polys = []
        for poly in imgaug_polys.polygons:
            new_poly = []
            for point in poly:
                new_poly.append(np.array(point, dtype=np.float32))
            new_poly = np.array(new_poly, dtype=np.float32).flatten()
            new_polys.append([new_poly])

        return new_polys

    def may_augment_poly_legacy(self, aug, img_shape, polys):
        key_points, poly_point_nums = [], []
        for poly in polys:
            poly = poly[0]
            poly = poly.reshape(-1, 2)
            key_points.extend([imgaug.Keypoint(p[0], p[1]) for p in poly])
            poly_point_nums.append(poly.shape[0])
        # Warning: we do not clip the out-of-boudnary polygons
        key_points = aug.augment_keypoints(
            [imgaug.KeypointsOnImage(keypoints=key_points,
                                     shape=img_shape)])[0].keypoints

        new_polys = []
        start_idx = 0
        for poly_point_num in poly_point_nums:
            new_poly = []
            for key_point in key_points[start_idx:(start_idx +
                                                   poly_point_num)]:
                new_poly.append([key_point.x, key_point.y])
            start_idx += poly_point_num
            new_poly = np.array(new_poly).flatten()
            new_polys.append([new_poly])

        return new_polys

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class EastRandomCrop:

    def __init__(self,
                 target_size=(640, 640),
                 max_tries=10,
                 min_crop_side_ratio=0.1):
        self.target_size = target_size
        self.max_tries = max_tries
        self.min_crop_side_ratio = min_crop_side_ratio

    def __call__(self, results):
        # sampling crop
        # crop image, boxes, masks
        img = results['img']
        crop_x, crop_y, crop_w, crop_h = self.crop_area(
            img, results['gt_masks'])
        scale_w = self.target_size[0] / crop_w
        scale_h = self.target_size[1] / crop_h
        scale = min(scale_w, scale_h)
        h = int(crop_h * scale)
        w = int(crop_w * scale)
        padded_img = np.zeros(
            (self.target_size[1], self.target_size[0], img.shape[2]),
            img.dtype)
        padded_img[:h, :w] = mmcv.imresize(
            img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], (w, h))

        # for bboxes
        for key in results['bbox_fields']:
            lines = []
            for box in results[key]:
                box = box.reshape(2, 2)
                poly = ((box - (crop_x, crop_y)) * scale)
                if not self.is_poly_outside_rect(poly, 0, 0, w, h):
                    lines.append(poly.flatten())
            results[key] = np.array(lines)
        # for masks
        for key in results['mask_fields']:
            polys = []
            polys_label = []
            for poly in results[key]:
                poly = np.array(poly).reshape(-1, 2)
                poly = ((poly - (crop_x, crop_y)) * scale)
                if not self.is_poly_outside_rect(poly, 0, 0, w, h):
                    polys.append([poly])
                    polys_label.append(0)
            results[key] = PolygonMasks(polys, *self.target_size)
            if key == 'gt_masks':
                results['gt_labels'] = polys_label

        results['img'] = padded_img
        results['img_shape'] = padded_img.shape

        return results

    def is_poly_in_rect(self, poly, x, y, w, h):
        poly = np.array(poly)
        if poly[:, 0].min() < x or poly[:, 0].max() > x + w:
            return False
        if poly[:, 1].min() < y or poly[:, 1].max() > y + h:
            return False
        return True

    def is_poly_outside_rect(self, poly, x, y, w, h):
        poly = np.array(poly).reshape(-1, 2)
        if poly[:, 0].max() < x or poly[:, 0].min() > x + w:
            return True
        if poly[:, 1].max() < y or poly[:, 1].min() > y + h:
            return True
        return False

    def split_regions(self, axis):
        regions = []
        min_axis = 0
        for i in range(1, axis.shape[0]):
            if axis[i] != axis[i - 1] + 1:
                region = axis[min_axis:i]
                min_axis = i
                regions.append(region)
        return regions

    def random_select(self, axis, max_size):
        xx = np.random.choice(axis, size=2)
        xmin = np.min(xx)
        xmax = np.max(xx)
        xmin = np.clip(xmin, 0, max_size - 1)
        xmax = np.clip(xmax, 0, max_size - 1)
        return xmin, xmax

    def region_wise_random_select(self, regions):
        selected_index = list(np.random.choice(len(regions), 2))
        selected_values = []
        for index in selected_index:
            axis = regions[index]
            xx = int(np.random.choice(axis, size=1))
            selected_values.append(xx)
        xmin = min(selected_values)
        xmax = max(selected_values)
        return xmin, xmax

    def crop_area(self, img, polys):
        h, w, _ = img.shape
        h_array = np.zeros(h, dtype=np.int32)
        w_array = np.zeros(w, dtype=np.int32)
        for points in polys:
            points = np.round(
                points, decimals=0).astype(np.int32).reshape(-1, 2)
            min_x = np.min(points[:, 0])
            max_x = np.max(points[:, 0])
            w_array[min_x:max_x] = 1
            min_y = np.min(points[:, 1])
            max_y = np.max(points[:, 1])
            h_array[min_y:max_y] = 1
        # ensure the cropped area not across a text
        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]

        if len(h_axis) == 0 or len(w_axis) == 0:
            return 0, 0, w, h

        h_regions = self.split_regions(h_axis)
        w_regions = self.split_regions(w_axis)

        for i in range(self.max_tries):
            if len(w_regions) > 1:
                xmin, xmax = self.region_wise_random_select(w_regions)
            else:
                xmin, xmax = self.random_select(w_axis, w)
            if len(h_regions) > 1:
                ymin, ymax = self.region_wise_random_select(h_regions)
            else:
                ymin, ymax = self.random_select(h_axis, h)

            if (xmax - xmin < self.min_crop_side_ratio * w
                    or ymax - ymin < self.min_crop_side_ratio * h):
                # area too small
                continue
            num_poly_in_rect = 0
            for poly in polys:
                if not self.is_poly_outside_rect(poly, xmin, ymin, xmax - xmin,
                                                 ymax - ymin):
                    num_poly_in_rect += 1
                    break

            if num_poly_in_rect > 0:
                return xmin, ymin, xmax - xmin, ymax - ymin

        return 0, 0, w, h

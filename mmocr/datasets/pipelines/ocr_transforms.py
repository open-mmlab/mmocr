import math

import mmcv
import numpy as np
import torch
import torchvision.transforms.functional as TF
from mmcv.runner.dist_utils import get_dist_info
from PIL import Image
from shapely.geometry import Polygon
from shapely.geometry import box as shapely_box

import mmocr.utils as utils
from mmdet.datasets.builder import PIPELINES
from mmocr.datasets.pipelines.crop import warp_img


@PIPELINES.register_module()
class ResizeOCR:
    """Image resizing and padding for OCR.

    Args:
        height (int | tuple(int)): Image height after resizing.
        min_width (none | int | tuple(int)): Image minimum width
            after resizing.
        max_width (none | int | tuple(int)): Image maximum width
            after resizing.
        keep_aspect_ratio (bool): Keep image aspect ratio if True
            during resizing, Otherwise resize to the size height *
            max_width.
        img_pad_value (int): Scalar to fill padding area.
        width_downsample_ratio (float): Downsample ratio in horizontal
            direction from input image to output feature.
        backend (str | None): The image resize backend type. Options are `cv2`,
            `pillow`, `None`. If backend is None, the global imread_backend
            specified by ``mmcv.use_backend()`` will be used. Default: None.
    """

    def __init__(self,
                 height,
                 min_width=None,
                 max_width=None,
                 keep_aspect_ratio=True,
                 img_pad_value=0,
                 width_downsample_ratio=1.0 / 16,
                 backend=None):
        assert isinstance(height, (int, tuple))
        assert utils.is_none_or_type(min_width, (int, tuple))
        assert utils.is_none_or_type(max_width, (int, tuple))
        if not keep_aspect_ratio:
            assert max_width is not None, ('"max_width" must assigned '
                                           'if "keep_aspect_ratio" is False')
        assert isinstance(img_pad_value, int)
        if isinstance(height, tuple):
            assert isinstance(min_width, tuple)
            assert isinstance(max_width, tuple)
            assert len(height) == len(min_width) == len(max_width)

        self.height = height
        self.min_width = min_width
        self.max_width = max_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.img_pad_value = img_pad_value
        self.width_downsample_ratio = width_downsample_ratio
        self.backend = backend

    def __call__(self, results):
        rank, _ = get_dist_info()
        if isinstance(self.height, int):
            dst_height = self.height
            dst_min_width = self.min_width
            dst_max_width = self.max_width
        else:
            # Multi-scale resize used in distributed training.
            # Choose one (height, width) pair for one rank id.

            idx = rank % len(self.height)
            dst_height = self.height[idx]
            dst_min_width = self.min_width[idx]
            dst_max_width = self.max_width[idx]

        img_shape = results['img_shape']
        ori_height, ori_width = img_shape[:2]
        valid_ratio = 1.0
        resize_shape = list(img_shape)
        pad_shape = list(img_shape)

        if self.keep_aspect_ratio:
            new_width = math.ceil(float(dst_height) / ori_height * ori_width)
            width_divisor = int(1 / self.width_downsample_ratio)
            # make sure new_width is an integral multiple of width_divisor.
            if new_width % width_divisor != 0:
                new_width = round(new_width / width_divisor) * width_divisor
            if dst_min_width is not None:
                new_width = max(dst_min_width, new_width)
            if dst_max_width is not None:
                valid_ratio = min(1.0, 1.0 * new_width / dst_max_width)
                resize_width = min(dst_max_width, new_width)
                img_resize = mmcv.imresize(
                    results['img'], (resize_width, dst_height),
                    backend=self.backend)
                resize_shape = img_resize.shape
                pad_shape = img_resize.shape
                if new_width < dst_max_width:
                    img_resize = mmcv.impad(
                        img_resize,
                        shape=(dst_height, dst_max_width),
                        pad_val=self.img_pad_value)
                    pad_shape = img_resize.shape
            else:
                img_resize = mmcv.imresize(
                    results['img'], (new_width, dst_height),
                    backend=self.backend)
                resize_shape = img_resize.shape
                pad_shape = img_resize.shape
        else:
            img_resize = mmcv.imresize(
                results['img'], (dst_max_width, dst_height),
                backend=self.backend)
            resize_shape = img_resize.shape
            pad_shape = img_resize.shape

        results['img'] = img_resize
        results['resize_shape'] = resize_shape
        results['pad_shape'] = pad_shape
        results['valid_ratio'] = valid_ratio

        return results


@PIPELINES.register_module()
class ToTensorOCR:
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor."""

    def __init__(self):
        pass

    def __call__(self, results):
        results['img'] = TF.to_tensor(results['img'].copy())

        return results


@PIPELINES.register_module()
class NormalizeOCR:
    """Normalize a tensor image with mean and standard deviation."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, results):
        results['img'] = TF.normalize(results['img'], self.mean, self.std)
        results['img_norm_cfg'] = dict(mean=self.mean, std=self.std)
        return results


@PIPELINES.register_module()
class OnlineCropOCR:
    """Crop text areas from whole image with bounding box jitter. If no bbox is
    given, return directly.

    Args:
        box_keys (list[str]): Keys in results which correspond to RoI bbox.
        jitter_prob (float): The probability of box jitter.
        max_jitter_ratio_x (float): Maximum horizontal jitter ratio
            relative to height.
        max_jitter_ratio_y (float): Maximum vertical jitter ratio
            relative to height.
    """

    def __init__(self,
                 box_keys=['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'],
                 jitter_prob=0.5,
                 max_jitter_ratio_x=0.05,
                 max_jitter_ratio_y=0.02):
        assert utils.is_type_list(box_keys, str)
        assert 0 <= jitter_prob <= 1
        assert 0 <= max_jitter_ratio_x <= 1
        assert 0 <= max_jitter_ratio_y <= 1

        self.box_keys = box_keys
        self.jitter_prob = jitter_prob
        self.max_jitter_ratio_x = max_jitter_ratio_x
        self.max_jitter_ratio_y = max_jitter_ratio_y

    def __call__(self, results):

        if 'img_info' not in results:
            return results

        crop_flag = True
        box = []
        for key in self.box_keys:
            if key not in results['img_info']:
                crop_flag = False
                break

            box.append(float(results['img_info'][key]))

        if not crop_flag:
            return results

        jitter_flag = np.random.random() > self.jitter_prob

        kwargs = dict(
            jitter_flag=jitter_flag,
            jitter_ratio_x=self.max_jitter_ratio_x,
            jitter_ratio_y=self.max_jitter_ratio_y)
        crop_img = warp_img(results['img'], box, **kwargs)

        results['img'] = crop_img
        results['img_shape'] = crop_img.shape

        return results


@PIPELINES.register_module()
class FancyPCA:
    """Implementation of PCA based image augmentation, proposed in the paper
    ``Imagenet Classification With Deep Convolutional Neural Networks``.

    It alters the intensities of RGB values along the principal components of
    ImageNet dataset.
    """

    def __init__(self, eig_vec=None, eig_val=None):
        if eig_vec is None:
            eig_vec = torch.Tensor([
                [-0.5675, +0.7192, +0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948, +0.4203],
            ]).t()
        if eig_val is None:
            eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])
        self.eig_val = eig_val  # 1*3
        self.eig_vec = eig_vec  # 3*3

    def pca(self, tensor):
        assert tensor.size(0) == 3
        alpha = torch.normal(mean=torch.zeros_like(self.eig_val)) * 0.1
        reconst = torch.mm(self.eig_val * alpha, self.eig_vec)
        tensor = tensor + reconst.view(3, 1, 1)

        return tensor

    def __call__(self, results):
        img = results['img']
        tensor = self.pca(img)
        results['img'] = tensor

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class RandomPaddingOCR:
    """Pad the given image on all sides, as well as modify the coordinates of
    character bounding box in image.

    Args:
        max_ratio (list[int]): [left, top, right, bottom].
        box_type (None|str): Character box type. If not none,
            should be either 'char_rects' or 'char_quads', with
            'char_rects' for rectangle with ``xyxy`` style and
            'char_quads' for quadrangle with ``x1y1x2y2x3y3x4y4`` style.
    """

    def __init__(self, max_ratio=None, box_type=None):
        if max_ratio is None:
            max_ratio = [0.1, 0.2, 0.1, 0.2]
        else:
            assert utils.is_type_list(max_ratio, float)
            assert len(max_ratio) == 4
        assert box_type is None or box_type in ('char_rects', 'char_quads')

        self.max_ratio = max_ratio
        self.box_type = box_type

    def __call__(self, results):

        img_shape = results['img_shape']
        ori_height, ori_width = img_shape[:2]

        random_padding_left = round(
            np.random.uniform(0, self.max_ratio[0]) * ori_width)
        random_padding_top = round(
            np.random.uniform(0, self.max_ratio[1]) * ori_height)
        random_padding_right = round(
            np.random.uniform(0, self.max_ratio[2]) * ori_width)
        random_padding_bottom = round(
            np.random.uniform(0, self.max_ratio[3]) * ori_height)

        padding = (random_padding_left, random_padding_top,
                   random_padding_right, random_padding_bottom)
        img = mmcv.impad(results['img'], padding=padding, padding_mode='edge')

        results['img'] = img
        results['img_shape'] = img.shape

        if self.box_type is not None:
            num_points = 2 if self.box_type == 'char_rects' else 4
            char_num = len(results['ann_info'][self.box_type])
            for i in range(char_num):
                for j in range(num_points):
                    results['ann_info'][self.box_type][i][
                        j * 2] += random_padding_left
                    results['ann_info'][self.box_type][i][
                        j * 2 + 1] += random_padding_top

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class RandomRotateImageBox:
    """Rotate augmentation for segmentation based text recognition.

    Args:
        min_angle (int): Minimum rotation angle for image and box.
        max_angle (int): Maximum rotation angle for image and box.
        box_type (str): Character box type, should be either
            'char_rects' or 'char_quads', with 'char_rects'
            for rectangle with ``xyxy`` style and 'char_quads'
            for quadrangle with ``x1y1x2y2x3y3x4y4`` style.
    """

    def __init__(self, min_angle=-10, max_angle=10, box_type='char_quads'):
        assert box_type in ('char_rects', 'char_quads')

        self.min_angle = min_angle
        self.max_angle = max_angle
        self.box_type = box_type

    def __call__(self, results):
        in_img = results['img']
        in_chars = results['ann_info']['chars']
        in_boxes = results['ann_info'][self.box_type]

        img_width, img_height = in_img.size
        rotate_center = [img_width / 2., img_height / 2.]

        tan_temp_max_angle = rotate_center[1] / rotate_center[0]
        temp_max_angle = np.arctan(tan_temp_max_angle) * 180. / np.pi

        random_angle = np.random.uniform(
            max(self.min_angle, -temp_max_angle),
            min(self.max_angle, temp_max_angle))
        random_angle_radian = random_angle * np.pi / 180.

        img_box = shapely_box(0, 0, img_width, img_height)

        out_img = TF.rotate(
            in_img,
            random_angle,
            resample=False,
            expand=False,
            center=rotate_center)

        out_boxes, out_chars = self.rotate_bbox(in_boxes, in_chars,
                                                random_angle_radian,
                                                rotate_center, img_box)

        results['img'] = out_img
        results['ann_info']['chars'] = out_chars
        results['ann_info'][self.box_type] = out_boxes

        return results

    @staticmethod
    def rotate_bbox(boxes, chars, angle, center, img_box):
        out_boxes = []
        out_chars = []
        for idx, bbox in enumerate(boxes):
            temp_bbox = []
            for i in range(len(bbox) // 2):
                point = [bbox[2 * i], bbox[2 * i + 1]]
                temp_bbox.append(
                    RandomRotateImageBox.rotate_point(point, angle, center))
            poly_temp_bbox = Polygon(temp_bbox).buffer(0)
            if poly_temp_bbox.is_valid:
                if img_box.intersects(poly_temp_bbox) and (
                        not img_box.touches(poly_temp_bbox)):
                    temp_bbox_area = poly_temp_bbox.area

                    intersect_area = img_box.intersection(poly_temp_bbox).area
                    intersect_ratio = intersect_area / temp_bbox_area

                    if intersect_ratio >= 0.7:
                        out_box = []
                        for p in temp_bbox:
                            out_box.extend(p)
                        out_boxes.append(out_box)
                        out_chars.append(chars[idx])

        return out_boxes, out_chars

    @staticmethod
    def rotate_point(point, angle, center):
        cos_theta = math.cos(-angle)
        sin_theta = math.sin(-angle)
        c_x = center[0]
        c_y = center[1]
        new_x = (point[0] - c_x) * cos_theta - (point[1] -
                                                c_y) * sin_theta + c_x
        new_y = (point[0] - c_x) * sin_theta + (point[1] -
                                                c_y) * cos_theta + c_y

        return [new_x, new_y]


@PIPELINES.register_module()
class OpencvToPil:
    """Convert ``numpy.ndarray`` (bgr) to ``PIL Image`` (rgb)."""

    def __init__(self, **kwargs):
        pass

    def __call__(self, results):
        img = results['img'][..., ::-1]
        img = Image.fromarray(img)
        results['img'] = img

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class PilToOpencv:
    """Convert ``PIL Image`` (rgb) to ``numpy.ndarray`` (bgr)."""

    def __init__(self, **kwargs):
        pass

    def __call__(self, results):
        img = np.asarray(results['img'])
        img = img[..., ::-1]
        results['img'] = img

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

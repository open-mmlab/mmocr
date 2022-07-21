# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from mmdet.models.utils import multi_apply
from shapely.geometry import Polygon
from torch import Tensor, nn

from mmocr.data import TextDetDataSample
from mmocr.registry import MODELS
from mmocr.utils import offset_polygon
from mmocr.utils.typing import ArrayLike
from .text_kernel_mixin import TextKernelMixin


@MODELS.register_module()
class DBModuleLoss(nn.Module, TextKernelMixin):
    r"""The class for implementing DBNet loss.

    This is partially adapted from https://github.com/MhLiao/DB.

    Args:
        loss_prob (dict): The loss config for probability map.
        loss_thr (dict): The loss config for threshold map.
        loss_db (dict): The loss config for binary map.
        weight_prob (float): The weight of probability map loss.
            Denoted as :math:`\alpha` in paper.
        weight_thr (float): The weight of threshold map loss.
            Denoted as :math:`\beta` in paper.
        shrink_ratio (float): The ratio of shrunk text region.
        thr_min (float): The minimum threshold map value.
        thr_max (float): The maximum threshold map value.
        min_sidelength (int or float): The minimum sidelength of the
            minimum rotated rectangle around any text region.
    """

    def __init__(self,
                 loss_prob: Dict = dict(type='MaskedBalancedBCELoss'),
                 loss_thr: Dict = dict(type='MaskedSmoothL1Loss', beta=0),
                 loss_db: Dict = dict(type='MaskedDiceLoss'),
                 weight_prob: float = 5.,
                 weight_thr: float = 10.,
                 shrink_ratio: float = 0.4,
                 thr_min: float = 0.3,
                 thr_max: float = 0.7,
                 min_sidelength: Union[int, float] = 8) -> None:
        super().__init__()
        self.loss_prob = MODELS.build(loss_prob)
        self.loss_thr = MODELS.build(loss_thr)
        self.loss_db = MODELS.build(loss_db)
        self.weight_prob = weight_prob
        self.weight_thr = weight_thr
        self.shrink_ratio = shrink_ratio
        self.thr_min = thr_min
        self.thr_max = thr_max
        self.min_sidelength = min_sidelength

    def forward(self, preds: Tuple[Tensor, Tensor, Tensor],
                data_samples: Sequence[TextDetDataSample]) -> Dict:
        """Compute DBNet loss.

        Args:
            preds (tuple(tensor)): Raw predictions from model, containing
                ``prob_map``, ``thr_map`` and ``binary_map``. Each is a tensor
                of shape :math:`(N, H, W)`.
            data_samples (list[TextDetDataSample]): The data samples.

        Returns:
            results(dict): The dict for dbnet losses with loss_prob, \
                loss_db and loss_thr.
        """
        prob_map, thr_map, binary_map = preds
        gt_shrinks, gt_shrink_masks, gt_thrs, gt_thr_masks = self.get_targets(
            data_samples)
        gt_shrinks = gt_shrinks.to(prob_map.device)
        gt_shrink_masks = gt_shrink_masks.to(prob_map.device)
        gt_thrs = gt_thrs.to(thr_map.device)
        gt_thr_masks = gt_thr_masks.to(thr_map.device)
        loss_prob = self.loss_prob(prob_map, gt_shrinks, gt_shrink_masks)

        loss_thr = self.loss_thr(thr_map, gt_thrs, gt_thr_masks)
        loss_db = self.loss_db(binary_map, gt_shrinks, gt_shrink_masks)

        results = dict(
            loss_prob=self.weight_prob * loss_prob,
            loss_thr=self.weight_thr * loss_thr,
            loss_db=loss_db)

        return results

    def _is_poly_invalid(self, poly: np.ndarray) -> bool:
        """Check if the input polygon is invalid or not. It is invalid if its
        area is smaller than 1 or the shorter side of its minimum bounding box
        is smaller than min_sidelength.

        Args:
            poly (ndarray): The polygon.

        Returns:
            bool: Whether the polygon is invalid.
        """
        poly = poly.reshape(-1, 2)
        area = Polygon(poly).area
        if abs(area) < 1:
            return True
        rect_size = cv2.minAreaRect(poly)[1]
        len_shortest_side = min(rect_size)
        if len_shortest_side < self.min_sidelength:
            return True

        return False

    def _generate_thr_map(self, img_size: Tuple[int, int],
                          polygons: ArrayLike) -> np.ndarray:
        """Generate threshold map.

        Args:
            img_size (tuple(int)): The image size (h, w)
            polygons (Sequence[ndarray]): 2-d array, representing all the
                polygons of the text region.

        Returns:
            tuple:

            - thr_map (ndarray): The generated threshold map.
            - thr_mask (ndarray): The effective mask of threshold map.
        """
        thr_map = np.zeros(img_size, dtype=np.float32)
        thr_mask = np.zeros(img_size, dtype=np.uint8)

        for polygon in polygons:
            self._draw_border_map(polygon, thr_map, mask=thr_mask)
        thr_map = thr_map * (self.thr_max - self.thr_min) + self.thr_min

        return thr_map, thr_mask

    def _draw_border_map(self, polygon: np.ndarray, canvas: np.ndarray,
                         mask: np.ndarray) -> None:
        """Generate threshold map for one polygon.

        Args:
            polygon (np.ndarray): The polygon.
            canvas (np.ndarray): The generated threshold map.
            mask (np.ndarray): The generated threshold mask.
        """

        polygon = polygon.reshape(-1, 2)
        polygon_obj = Polygon(polygon)
        distance = (
            polygon_obj.area * (1 - np.power(self.shrink_ratio, 2)) /
            polygon_obj.length)
        expanded_polygon = offset_polygon(polygon, distance)
        if len(expanded_polygon) == 0:
            print(f'Padding {polygon} with {distance} gets {expanded_polygon}')
            expanded_polygon = polygon.copy().astype(np.int32)
        else:
            expanded_polygon = expanded_polygon.reshape(-1, 2).astype(np.int32)

        x_min = expanded_polygon[:, 0].min()
        x_max = expanded_polygon[:, 0].max()
        y_min = expanded_polygon[:, 1].min()
        y_max = expanded_polygon[:, 1].max()

        width = x_max - x_min + 1
        height = y_max - y_min + 1

        polygon[:, 0] = polygon[:, 0] - x_min
        polygon[:, 1] = polygon[:, 1] - y_min

        xs = np.broadcast_to(
            np.linspace(0, width - 1, num=width).reshape(1, width),
            (height, width))
        ys = np.broadcast_to(
            np.linspace(0, height - 1, num=height).reshape(height, 1),
            (height, width))

        distance_map = np.zeros((polygon.shape[0], height, width),
                                dtype=np.float32)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            absolute_distance = self._dist_points2line(xs, ys, polygon[i],
                                                       polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = distance_map.min(axis=0)

        x_min_valid = min(max(0, x_min), canvas.shape[1] - 1)
        x_max_valid = min(max(0, x_max), canvas.shape[1] - 1)
        y_min_valid = min(max(0, y_min), canvas.shape[0] - 1)
        y_max_valid = min(max(0, y_max), canvas.shape[0] - 1)

        if x_min_valid - x_min >= width or y_min_valid - y_min >= height:
            return

        cv2.fillPoly(mask, [expanded_polygon.astype(np.int32)], 1.0)
        canvas[y_min_valid:y_max_valid + 1,
               x_min_valid:x_max_valid + 1] = np.fmax(
                   1 - distance_map[y_min_valid - y_min:y_max_valid - y_max +
                                    height, x_min_valid - x_min:x_max_valid -
                                    x_max + width],
                   canvas[y_min_valid:y_max_valid + 1,
                          x_min_valid:x_max_valid + 1])

    def get_targets(self, data_samples: List[TextDetDataSample]) -> Tuple:
        """Generate loss targets from data samples.

        Args:
            data_samples (list(TextDetDataSample)): Ground truth data samples.

        Returns:
            tuple: A tuple of four tensors as DBNet targets.
        """

        gt_shrinks, gt_shrink_masks, gt_thrs, gt_thr_masks = multi_apply(
            self._get_target_single, data_samples)
        gt_shrinks = torch.cat(gt_shrinks)
        gt_shrink_masks = torch.cat(gt_shrink_masks)
        gt_thrs = torch.cat(gt_thrs)
        gt_thr_masks = torch.cat(gt_thr_masks)
        return gt_shrinks, gt_shrink_masks, gt_thrs, gt_thr_masks

    def _get_target_single(self, data_sample: TextDetDataSample) -> Tuple:
        """Generate loss target from a data sample.

        Args:
            data_sample (TextDetDataSample): The data sample.

        Returns:
            tuple: A tuple of four tensors as the targets of one prediction.
        """

        gt_instances = data_sample.gt_instances
        ignore_flags = gt_instances.ignored
        for idx, polygon in enumerate(gt_instances.polygons):
            if self._is_poly_invalid(polygon):
                ignore_flags[idx] = True
        gt_shrink, ignore_flags = self._generate_kernels(
            data_sample.img_shape,
            gt_instances.polygons,
            self.shrink_ratio,
            ignore_flags=ignore_flags)

        # Get boolean mask where Trues indicate text instance pixels
        gt_shrink = gt_shrink > 0

        gt_shrink_mask = self._generate_effective_mask(
            data_sample.img_shape, gt_instances[ignore_flags].polygons)
        gt_thr, gt_thr_mask = self._generate_thr_map(
            data_sample.img_shape, gt_instances[~ignore_flags].polygons)

        # to_tensor
        gt_shrink = torch.from_numpy(gt_shrink).unsqueeze(0).float()
        gt_shrink_mask = torch.from_numpy(gt_shrink_mask).unsqueeze(0).float()
        gt_thr = torch.from_numpy(gt_thr).unsqueeze(0).float()
        gt_thr_mask = torch.from_numpy(gt_thr_mask).unsqueeze(0).float()
        return gt_shrink, gt_shrink_mask, gt_thr, gt_thr_mask

    @staticmethod
    def _dist_points2line(xs: np.ndarray, ys: np.ndarray, pt1: np.ndarray,
                          pt2: np.ndarray) -> np.ndarray:
        """Compute distances from points to a line. This is adapted from
        https://github.com/MhLiao/DB.

        Args:
            xs (ndarray): The x coordinates of points of size :math:`(N, )`.
            ys (ndarray): The y coordinates of size :math:`(N, )`.
            pt1 (ndarray): The first point on the line of size :math:`(2, )`.
            pt2 (ndarray): The second point on the line of size :math:`(2, )`.

        Returns:
            ndarray: The distance matrix of size :math:`(N, )`.
        """
        # suppose a triangle with three edge abc with c=point_1 point_2
        # a^2
        a_square = np.square(xs - pt1[0]) + np.square(ys - pt1[1])
        # b^2
        b_square = np.square(xs - pt2[0]) + np.square(ys - pt2[1])
        # c^2
        c_square = np.square(pt1[0] - pt2[0]) + np.square(pt1[1] - pt2[1])
        # -cosC=(c^2-a^2-b^2)/2(ab)
        neg_cos_c = (
            (c_square - a_square - b_square) /
            (np.finfo(np.float32).eps + 2 * np.sqrt(a_square * b_square)))
        # sinC^2=1-cosC^2
        square_sin = 1 - np.square(neg_cos_c)
        square_sin = np.nan_to_num(square_sin)
        # distance=a*b*sinC/c=a*h/c=2*area/c
        result = np.sqrt(a_square * b_square * square_sin /
                         (np.finfo(np.float32).eps + c_square))
        # set result to minimum edge if C<pi/2
        result[neg_cos_c < 0] = np.sqrt(np.fmin(a_square,
                                                b_square))[neg_cos_c < 0]
        return result

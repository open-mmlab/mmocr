# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from mmdet.models.utils import multi_apply
from numpy.fft import fft
from numpy.linalg import norm

from mmocr.registry import MODELS
from mmocr.structures import TextDetDataSample
from mmocr.utils.typing_utils import ArrayLike
from .textsnake_module_loss import TextSnakeModuleLoss


@MODELS.register_module()
class FCEModuleLoss(TextSnakeModuleLoss):
    """The class for implementing FCENet loss.

    FCENet(CVPR2021): `Fourier Contour Embedding for Arbitrary-shaped Text
    Detection <https://arxiv.org/abs/2104.10442>`_

    Args:
        fourier_degree (int) : The maximum Fourier transform degree k.
        num_sample (int) : The sampling points number of regression
            loss. If it is too small, fcenet tends to be overfitting.
        negative_ratio (float or int): Maximum ratio of negative
            samples to positive ones in OHEM. Defaults to 3.
        resample_step (float): The step size for resampling the text center
            line (TCL). It's better not to exceed half of the minimum width.
        center_region_shrink_ratio (float): The shrink ratio of text center
            region.
        level_size_divisors (tuple(int)): The downsample ratio on each level.
        level_proportion_range (tuple(tuple(int))): The range of text sizes
            assigned to each level.
        loss_tr (dict) : The loss config used to calculate the text region
            loss. Defaults to dict(type='MaskedBalancedBCELoss').
        loss_tcl (dict) : The loss config used to calculate the text center
            line loss. Defaults to dict(type='MaskedBCELoss').
        loss_reg_x (dict) : The loss config used to calculate the regression
            loss on x axis. Defaults to dict(type='MaskedSmoothL1Loss').
        loss_reg_y (dict) : The loss config used to calculate the regression
            loss on y axis. Defaults to dict(type='MaskedSmoothL1Loss').
    """

    def __init__(
        self,
        fourier_degree: int,
        num_sample: int,
        negative_ratio: Union[float, int] = 3.,
        resample_step: float = 4.0,
        center_region_shrink_ratio: float = 0.3,
        level_size_divisors: Tuple[int] = (8, 16, 32),
        level_proportion_range: Tuple[Tuple[int]] = ((0, 0.4), (0.3, 0.7),
                                                     (0.6, 1.0)),
        loss_tr: Dict = dict(type='MaskedBalancedBCELoss'),
        loss_tcl: Dict = dict(type='MaskedBCELoss'),
        loss_reg_x: Dict = dict(type='SmoothL1Loss', reduction='none'),
        loss_reg_y: Dict = dict(type='SmoothL1Loss', reduction='none'),
    ) -> None:
        super().__init__()
        self.fourier_degree = fourier_degree
        self.num_sample = num_sample
        self.resample_step = resample_step
        self.center_region_shrink_ratio = center_region_shrink_ratio
        self.level_size_divisors = level_size_divisors
        self.level_proportion_range = level_proportion_range

        loss_tr.update(negative_ratio=negative_ratio)
        self.loss_tr = MODELS.build(loss_tr)
        self.loss_tcl = MODELS.build(loss_tcl)
        self.loss_reg_x = MODELS.build(loss_reg_x)
        self.loss_reg_y = MODELS.build(loss_reg_y)

    def forward(self, preds: Sequence[Dict],
                data_samples: Sequence[TextDetDataSample]) -> Dict:
        """Compute FCENet loss.

        Args:
            preds (list[dict]): A list of dict with keys of ``cls_res``,
                ``reg_res`` corresponds to the classification result and
                regression result computed from the input tensor with the
                same index. They have the shapes of :math:`(N, C_{cls,i}, H_i,
                W_i)` and :math: `(N, C_{out,i}, H_i, W_i)`.
            data_samples (list[TextDetDataSample]): The data samples.

        Returns:
            dict: The dict for fcenet losses with loss_text, loss_center,
                loss_reg_x and loss_reg_y.
        """
        assert isinstance(preds, list) and len(preds) == 3
        p3_maps, p4_maps, p5_maps = self.get_targets(data_samples)
        device = preds[0]['cls_res'].device
        # to device
        gts = [p3_maps.to(device), p4_maps.to(device), p5_maps.to(device)]

        losses = multi_apply(self.forward_single, preds, gts)

        loss_tr = torch.tensor(0., device=device).float()
        loss_tcl = torch.tensor(0., device=device).float()
        loss_reg_x = torch.tensor(0., device=device).float()
        loss_reg_y = torch.tensor(0., device=device).float()

        for idx, loss in enumerate(losses):
            if idx == 0:
                loss_tr += sum(loss)
            elif idx == 1:
                loss_tcl += sum(loss)
            elif idx == 2:
                loss_reg_x += sum(loss)
            else:
                loss_reg_y += sum(loss)

        results = dict(
            loss_text=loss_tr,
            loss_center=loss_tcl,
            loss_reg_x=loss_reg_x,
            loss_reg_y=loss_reg_y,
        )

        return results

    def forward_single(self, pred: torch.Tensor,
                       gt: torch.Tensor) -> Sequence[torch.Tensor]:
        """Compute loss for one feature level.

        Args:
            pred (dict): A dict with keys ``cls_res`` and ``reg_res``
                corresponds to the classification result and regression result
                from one feature level.
            gt (Tensor): Ground truth for one feature level. Cls and reg
                targets are concatenated along the channel dimension.

        Returns:
            list[Tensor]: A list of losses for each feature level.
        """
        assert isinstance(pred, dict) and isinstance(gt, torch.Tensor)
        cls_pred = pred['cls_res'].permute(0, 2, 3, 1).contiguous()
        reg_pred = pred['reg_res'].permute(0, 2, 3, 1).contiguous()

        gt = gt.permute(0, 2, 3, 1).contiguous()

        k = 2 * self.fourier_degree + 1
        tr_pred = cls_pred[:, :, :, :2].view(-1, 2)
        tcl_pred = cls_pred[:, :, :, 2:].view(-1, 2)
        x_pred = reg_pred[:, :, :, 0:k].view(-1, k)
        y_pred = reg_pred[:, :, :, k:2 * k].view(-1, k)

        tr_mask = gt[:, :, :, :1].view(-1)
        tcl_mask = gt[:, :, :, 1:2].view(-1)
        train_mask = gt[:, :, :, 2:3].view(-1)
        x_map = gt[:, :, :, 3:3 + k].view(-1, k)
        y_map = gt[:, :, :, 3 + k:].view(-1, k)

        tr_train_mask = (train_mask * tr_mask).float()
        # text region loss
        loss_tr = self.loss_tr(tr_pred.softmax(-1)[:, 1], tr_mask, train_mask)

        # text center line loss
        tr_neg_mask = 1 - tr_train_mask
        loss_tcl_positive = self.loss_center(
            tcl_pred.softmax(-1)[:, 1], tcl_mask, tr_train_mask)
        loss_tcl_negative = self.loss_center(
            tcl_pred.softmax(-1)[:, 1], tcl_mask, tr_neg_mask)
        loss_tcl = loss_tcl_positive + 0.5 * loss_tcl_negative

        # regression loss
        loss_reg_x = torch.tensor(0.).float().to(x_pred.device)
        loss_reg_y = torch.tensor(0.).float().to(x_pred.device)
        if tr_train_mask.sum().item() > 0:
            weight = (tr_mask[tr_train_mask.bool()].float() +
                      tcl_mask[tr_train_mask.bool()].float()) / 2
            weight = weight.contiguous().view(-1, 1)

            ft_x, ft_y = self._fourier2poly(x_map, y_map)
            ft_x_pre, ft_y_pre = self._fourier2poly(x_pred, y_pred)

            loss_reg_x = torch.mean(weight * self.loss_reg_x(
                ft_x_pre[tr_train_mask.bool()], ft_x[tr_train_mask.bool()]))
            loss_reg_y = torch.mean(weight * self.loss_reg_x(
                ft_y_pre[tr_train_mask.bool()], ft_y[tr_train_mask.bool()]))

        return loss_tr, loss_tcl, loss_reg_x, loss_reg_y

    def get_targets(self, data_samples: List[TextDetDataSample]) -> Tuple:
        """Generate loss targets for fcenet from data samples.

        Args:
            data_samples (list(TextDetDataSample)): Ground truth data samples.

        Returns:
            tuple[Tensor]: A tuple of three tensors from three different
                feature level as FCENet targets.
        """
        p3_maps, p4_maps, p5_maps = multi_apply(self._get_target_single,
                                                data_samples)
        p3_maps = torch.cat(p3_maps, 0)
        p4_maps = torch.cat(p4_maps, 0)
        p5_maps = torch.cat(p5_maps, 0)

        return p3_maps, p4_maps, p5_maps

    def _get_target_single(self, data_sample: TextDetDataSample) -> Tuple:
        """Generate loss target for fcenet from a data sample.

        Args:
            data_sample (TextDetDataSample): The data sample.

        Returns:
            tuple[Tensor]: A tuple of three tensors from three different
            feature level as the targets of one prediction.
        """
        img_size = data_sample.batch_input_shape[:2]
        text_polys = data_sample.gt_instances.polygons
        ignore_flags = data_sample.gt_instances.ignored

        p3_map, p4_map, p5_map = self._generate_level_targets(
            img_size, text_polys, ignore_flags)
        # to tesnor
        p3_map = torch.from_numpy(p3_map).unsqueeze(0).float()
        p4_map = torch.from_numpy(p4_map).unsqueeze(0).float()
        p5_map = torch.from_numpy(p5_map).unsqueeze(0).float()
        return p3_map, p4_map, p5_map

    def _generate_level_targets(self,
                                img_size: Tuple[int, int],
                                text_polys: List[ArrayLike],
                                ignore_flags: Optional[torch.BoolTensor] = None
                                ) -> Tuple[torch.Tensor]:
        """Generate targets for one feature level.

        Args:
            img_size (tuple(int, int)): The image size of (height, width).
            text_polys (List[ndarray]): 2D array of text polygons.
            ignore_flags (torch.BoolTensor, optional): Indicate whether the
                corresponding text polygon is ignored. Defaults to None.

        Returns:
            tuple[Tensor]: A tuple of three tensors from one feature level
            as the targets.
        """
        h, w = img_size
        lv_size_divs = self.level_size_divisors
        lv_proportion_range = self.level_proportion_range

        lv_size_divs = self.level_size_divisors
        lv_proportion_range = self.level_proportion_range
        lv_text_polys = [[] for i in range(len(lv_size_divs))]
        lv_ignore_polys = [[] for i in range(len(lv_size_divs))]
        level_maps = []

        for poly_ind, poly in enumerate(text_polys):
            poly = np.array(poly, dtype=np.int_).reshape((1, -1, 2))
            _, _, box_w, box_h = cv2.boundingRect(poly)
            proportion = max(box_h, box_w) / (h + 1e-8)

            for ind, proportion_range in enumerate(lv_proportion_range):
                if proportion_range[0] < proportion < proportion_range[1]:
                    if ignore_flags is not None and ignore_flags[poly_ind]:
                        lv_ignore_polys[ind].append(poly[0] /
                                                    lv_size_divs[ind])
                    else:
                        lv_text_polys[ind].append(poly[0] / lv_size_divs[ind])

        for ind, size_divisor in enumerate(lv_size_divs):
            current_level_maps = []
            level_img_size = (h // size_divisor, w // size_divisor)

            text_region = self._generate_text_region_mask(
                level_img_size, lv_text_polys[ind])[None]
            current_level_maps.append(text_region)

            center_region = self._generate_center_region_mask(
                level_img_size, lv_text_polys[ind])[None]
            current_level_maps.append(center_region)

            effective_mask = self._generate_effective_mask(
                level_img_size, lv_ignore_polys[ind])[None]
            current_level_maps.append(effective_mask)

            fourier_real_map, fourier_image_maps = self._generate_fourier_maps(
                level_img_size, lv_text_polys[ind])
            current_level_maps.append(fourier_real_map)
            current_level_maps.append(fourier_image_maps)

            level_maps.append(np.concatenate(current_level_maps))

        return level_maps

    def _generate_center_region_mask(self, img_size: Tuple[int, int],
                                     text_polys: ArrayLike) -> np.ndarray:
        """Generate text center region mask.

        Args:
            img_size (tuple): The image size of (height, width).
            text_polys (list[ndarray]): The list of text polygons.

        Returns:
            ndarray: The text center region mask.
        """

        assert isinstance(img_size, tuple)

        h, w = img_size

        center_region_mask = np.zeros((h, w), np.uint8)

        center_region_boxes = []
        for poly in text_polys:
            polygon_points = poly.reshape(-1, 2)
            _, _, top_line, bot_line = self._reorder_poly_edge(polygon_points)
            resampled_top_line, resampled_bot_line = self._resample_sidelines(
                top_line, bot_line, self.resample_step)
            resampled_bot_line = resampled_bot_line[::-1]
            center_line = (resampled_top_line + resampled_bot_line) / 2

            line_head_shrink_len = norm(resampled_top_line[0] -
                                        resampled_bot_line[0]) / 4.0
            line_tail_shrink_len = norm(resampled_top_line[-1] -
                                        resampled_bot_line[-1]) / 4.0
            head_shrink_num = int(line_head_shrink_len // self.resample_step)
            tail_shrink_num = int(line_tail_shrink_len // self.resample_step)
            if len(center_line) > head_shrink_num + tail_shrink_num + 2:
                center_line = center_line[head_shrink_num:len(center_line) -
                                          tail_shrink_num]
                resampled_top_line = resampled_top_line[
                    head_shrink_num:len(resampled_top_line) - tail_shrink_num]
                resampled_bot_line = resampled_bot_line[
                    head_shrink_num:len(resampled_bot_line) - tail_shrink_num]

            for i in range(0, len(center_line) - 1):
                tl = center_line[i] + (resampled_top_line[i] - center_line[i]
                                       ) * self.center_region_shrink_ratio
                tr = center_line[i + 1] + (
                    resampled_top_line[i + 1] -
                    center_line[i + 1]) * self.center_region_shrink_ratio
                br = center_line[i + 1] + (
                    resampled_bot_line[i + 1] -
                    center_line[i + 1]) * self.center_region_shrink_ratio
                bl = center_line[i] + (resampled_bot_line[i] - center_line[i]
                                       ) * self.center_region_shrink_ratio
                current_center_box = np.vstack([tl, tr, br,
                                                bl]).astype(np.int32)
                center_region_boxes.append(current_center_box)

        cv2.fillPoly(center_region_mask, center_region_boxes, 1)
        return center_region_mask

    def _generate_fourier_maps(self, img_size: Tuple[int, int],
                               text_polys: ArrayLike
                               ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate Fourier coefficient maps.

        Args:
            img_size (tuple): The image size of (height, width).
            text_polys (list[ndarray]): The list of text polygons.

        Returns:
            tuple(ndarray, ndarray):

            - fourier_real_map (ndarray): The Fourier coefficient real part
                maps.
            - fourier_image_map (ndarray): The Fourier coefficient image part
                maps.
        """

        assert isinstance(img_size, tuple)

        h, w = img_size
        k = self.fourier_degree
        real_map = np.zeros((k * 2 + 1, h, w), dtype=np.float32)
        imag_map = np.zeros((k * 2 + 1, h, w), dtype=np.float32)

        for poly in text_polys:
            mask = np.zeros((h, w), dtype=np.uint8)
            polygon = np.array(poly).reshape((1, -1, 2))
            cv2.fillPoly(mask, polygon.astype(np.int32), 1)
            fourier_coeff = self._cal_fourier_signature(polygon[0], k)
            for i in range(-k, k + 1):
                if i != 0:
                    real_map[i + k, :, :] = mask * fourier_coeff[i + k, 0] + (
                        1 - mask) * real_map[i + k, :, :]
                    imag_map[i + k, :, :] = mask * fourier_coeff[i + k, 1] + (
                        1 - mask) * imag_map[i + k, :, :]
                else:
                    yx = np.argwhere(mask > 0.5)
                    k_ind = np.ones((len(yx)), dtype=np.int64) * k
                    y, x = yx[:, 0], yx[:, 1]
                    real_map[k_ind, y, x] = fourier_coeff[k, 0] - x
                    imag_map[k_ind, y, x] = fourier_coeff[k, 1] - y

        return real_map, imag_map

    def _cal_fourier_signature(self, polygon: ArrayLike,
                               fourier_degree: int) -> np.ndarray:
        """Calculate Fourier signature from input polygon.

        Args:
              polygon (list[ndarray]): The input polygon.
              fourier_degree (int): The maximum Fourier degree K.
        Returns:
              ndarray: An array shaped (2k+1, 2) containing
                  real part and image part of 2k+1 Fourier coefficients.
        """
        resampled_polygon = self._resample_polygon(polygon)
        resampled_polygon = self._normalize_polygon(resampled_polygon)

        fourier_coeff = self._poly2fourier(resampled_polygon, fourier_degree)
        fourier_coeff = self._clockwise(fourier_coeff, fourier_degree)

        real_part = np.real(fourier_coeff).reshape((-1, 1))
        image_part = np.imag(fourier_coeff).reshape((-1, 1))
        fourier_signature = np.hstack([real_part, image_part])

        return fourier_signature

    def _resample_polygon(self,
                          polygon: ArrayLike,
                          n: int = 400) -> np.ndarray:
        """Resample one polygon with n points on its boundary.

        Args:
            polygon (list[ndarray]): The input polygon.
            n (int): The number of resampled points. Defaults to 400.
        Returns:
            ndarray: The resampled polygon.
        """
        length = []

        for i in range(len(polygon)):
            p1 = polygon[i]
            if i == len(polygon) - 1:
                p2 = polygon[0]
            else:
                p2 = polygon[i + 1]
            length.append(((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5)

        total_length = sum(length)
        n_on_each_line = (np.array(length) / (total_length + 1e-8)) * n
        n_on_each_line = n_on_each_line.astype(np.int32)
        new_polygon = []

        for i in range(len(polygon)):
            num = n_on_each_line[i]
            p1 = polygon[i]
            if i == len(polygon) - 1:
                p2 = polygon[0]
            else:
                p2 = polygon[i + 1]

            if num == 0:
                continue

            dxdy = (p2 - p1) / num
            for j in range(num):
                point = p1 + dxdy * j
                new_polygon.append(point)

        return np.array(new_polygon)

    def _normalize_polygon(self, polygon: ArrayLike) -> np.ndarray:
        """Normalize one polygon so that its start point is at right most.

        Args:
            polygon (list[ndarray]): The origin polygon.
        Returns:
            ndarray: The polygon with start point at right.
        """
        temp_polygon = polygon - polygon.mean(axis=0)
        x = np.abs(temp_polygon[:, 0])
        y = temp_polygon[:, 1]
        index_x = np.argsort(x)
        index_y = np.argmin(y[index_x[:8]])
        index = index_x[index_y]
        new_polygon = np.concatenate([polygon[index:], polygon[:index]])
        return new_polygon

    def _clockwise(self, fourier_coeff: np.ndarray,
                   fourier_degree: int) -> np.ndarray:
        """Make sure the polygon reconstructed from Fourier coefficients c in
        the clockwise direction.

        Args:
            fourier_coeff (ndarray[complex]): The Fourier coefficients.
            fourier_degree: The maximum Fourier degree K.
        Returns:
            lost[float]: The polygon in clockwise point order.
        """
        if np.abs(fourier_coeff[fourier_degree + 1]) > np.abs(
                fourier_coeff[fourier_degree - 1]):
            return fourier_coeff
        elif np.abs(fourier_coeff[fourier_degree + 1]) < np.abs(
                fourier_coeff[fourier_degree - 1]):
            return fourier_coeff[::-1]
        else:
            if np.abs(fourier_coeff[fourier_degree + 2]) > np.abs(
                    fourier_coeff[fourier_degree - 2]):
                return fourier_coeff
            else:
                return fourier_coeff[::-1]

    def _poly2fourier(self, polygon: ArrayLike,
                      fourier_degree: int) -> np.ndarray:
        """Perform Fourier transformation to generate Fourier coefficients ck
        from polygon.

        Args:
            polygon (list[ndarray]): An input polygon.
            fourier_degree (int): The maximum Fourier degree K.
        Returns:
            ndarray: Fourier coefficients.
        """
        points = polygon[:, 0] + polygon[:, 1] * 1j
        c_fft = fft(points) / len(points)
        c = np.hstack((c_fft[-fourier_degree:], c_fft[:fourier_degree + 1]))
        return c

    def _fourier2poly(self, real_maps: torch.Tensor,
                      imag_maps: torch.Tensor) -> Sequence[torch.Tensor]:
        """Transform Fourier coefficient maps to polygon maps.

        Args:
            real_maps (tensor): A map composed of the real parts of the
                Fourier coefficients, whose shape is (-1, 2k+1)
            imag_maps (tensor):A map composed of the imag parts of the
                Fourier coefficients, whose shape is (-1, 2k+1)

        Returns
            tuple(tensor, tensor):

            - x_maps (tensor): A map composed of the x value of the polygon
                represented by n sample points (xn, yn), whose shape is (-1, n)
            - y_maps (tensor): A map composed of the y value of the polygon
                represented by n sample points (xn, yn), whose shape is (-1, n)
        """

        device = real_maps.device

        k_vect = torch.arange(
            -self.fourier_degree,
            self.fourier_degree + 1,
            dtype=torch.float,
            device=device).view(-1, 1)
        i_vect = torch.arange(
            0, self.num_sample, dtype=torch.float, device=device).view(1, -1)

        transform_matrix = 2 * np.pi / self.num_sample * torch.mm(
            k_vect, i_vect)

        x1 = torch.einsum('ak, kn-> an', real_maps,
                          torch.cos(transform_matrix))
        x2 = torch.einsum('ak, kn-> an', imag_maps,
                          torch.sin(transform_matrix))
        y1 = torch.einsum('ak, kn-> an', real_maps,
                          torch.sin(transform_matrix))
        y2 = torch.einsum('ak, kn-> an', imag_maps,
                          torch.cos(transform_matrix))

        x_maps = x1 - x2
        y_maps = y1 + y2

        return x_maps, y_maps

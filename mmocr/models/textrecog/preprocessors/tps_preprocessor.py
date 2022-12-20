# Copyright (c) OpenMMLab. All rights reserved.
import itertools
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmocr.registry import MODELS
from .base import BasePreprocessor


class TPStransform(nn.Module):
    """Implement TPS transform.

    This was partially adapted from https://github.com/ayumiymk/aster.pytorch

    Args:
        output_image_size (tuple[int, int]): The size of the output image.
            Defaults to (32, 128).
        num_control_points (int): The number of control points. Defaults to 20.
        margins (tuple[float, float]): The margins for control points to the
            top and down side of the image. Defaults to [0.05, 0.05].
    """

    def __init__(self,
                 output_image_size: Tuple[int, int] = (32, 100),
                 num_control_points: int = 20,
                 margins: Tuple[float, float] = [0.05, 0.05]) -> None:
        super().__init__()
        self.output_image_size = output_image_size
        self.num_control_points = num_control_points
        self.margins = margins
        self.target_height, self.target_width = output_image_size

        # build output control points
        target_control_points = self._build_output_control_points(
            num_control_points, margins)
        N = num_control_points

        # create padded kernel matrix
        forward_kernel = torch.zeros(N + 3, N + 3)
        target_control_partial_repr = self._compute_partial_repr(
            target_control_points, target_control_points)
        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        forward_kernel[:N, -3].fill_(1)
        forward_kernel[-3, :N].fill_(1)
        forward_kernel[:N, -2:].copy_(target_control_points)
        forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))

        # compute inverse matrix
        inverse_kernel = torch.inverse(forward_kernel).contiguous()

        # create target coordinate matrix
        HW = self.target_height * self.target_width
        tgt_coord = list(
            itertools.product(
                range(self.target_height), range(self.target_width)))
        tgt_coord = torch.Tensor(tgt_coord)
        Y, X = tgt_coord.split(1, dim=1)
        Y = Y / (self.target_height - 1)
        X = X / (self.target_width - 1)
        tgt_coord = torch.cat([X, Y], dim=1)
        tgt_coord_partial_repr = self._compute_partial_repr(
            tgt_coord, target_control_points)
        tgt_coord_repr = torch.cat(
            [tgt_coord_partial_repr,
             torch.ones(HW, 1), tgt_coord], dim=1)

        # register precomputed matrices
        self.register_buffer('inverse_kernel', inverse_kernel)
        self.register_buffer('padding_matrix', torch.zeros(3, 2))
        self.register_buffer('target_coordinate_repr', tgt_coord_repr)
        self.register_buffer('target_control_points', target_control_points)

    def forward(self, input: torch.Tensor,
                source_control_points: torch.Tensor) -> torch.Tensor:
        """Forward function of the TPS block.

        Args:
            input (Tensor): The input image.
            source_control_points (Tensor): The control points of the source
                image of shape (N, self.num_control_points, 2).
        Returns:
            Tensor: The output image after TPS transform.
        """
        assert source_control_points.ndimension() == 3
        assert source_control_points.size(1) == self.num_control_points
        assert source_control_points.size(2) == 2
        batch_size = source_control_points.size(0)

        Y = torch.cat([
            source_control_points,
            self.padding_matrix.expand(batch_size, 3, 2)
        ], 1)
        mapping_matrix = torch.matmul(self.inverse_kernel, Y)
        source_coordinate = torch.matmul(self.target_coordinate_repr,
                                         mapping_matrix)

        grid = source_coordinate.view(-1, self.target_height,
                                      self.target_width, 2)
        grid = torch.clamp(grid, 0, 1)
        grid = 2.0 * grid - 1.0
        output_maps = self._grid_sample(input, grid, canvas=None)
        return output_maps

    def _grid_sample(self,
                     input: torch.Tensor,
                     grid: torch.Tensor,
                     canvas: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample the input image at the given grid.

        Args:
            input (Tensor): The input image.
            grid (Tensor): The grid to sample the input image.
            canvas (Optional[Tensor]): The canvas to store the output image.
        Returns:
            Tensor: The sampled image.
        """
        output = F.grid_sample(input, grid, align_corners=True)
        if canvas is None:
            return output
        else:
            input_mask = input.data.new(input.size()).fill_(1)
            output_mask = F.grid_sample(input_mask, grid, align_corners=True)
            padded_output = output * output_mask + canvas * (1 - output_mask)
            return padded_output

    def _compute_partial_repr(self, input_points: torch.Tensor,
                              control_points: torch.Tensor) -> torch.Tensor:
        """Compute the partial representation matrix.

        Args:
            input_points (Tensor): The input points.
            control_points (Tensor): The control points.
        Returns:
            Tensor: The partial representation matrix.
        """
        N = input_points.size(0)
        M = control_points.size(0)
        pairwise_diff = input_points.view(N, 1, 2) - control_points.view(
            1, M, 2)
        pairwise_diff_square = pairwise_diff * pairwise_diff
        pairwise_dist = pairwise_diff_square[:, :,
                                             0] + pairwise_diff_square[:, :, 1]
        repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
        mask = repr_matrix != repr_matrix
        repr_matrix.masked_fill_(mask, 0)
        return repr_matrix

    # output_ctrl_pts are specified, according to our task.
    def _build_output_control_points(self, num_control_points: torch.Tensor,
                                     margins: Tuple[float,
                                                    float]) -> torch.Tensor:
        """Build the output control points.

        The output points will be fix at
        top and down side of the image.
        Args:
            num_control_points (Tensor): The number of control points.
            margins (Tuple[float, float]): The margins for control points to
                the top and down side of the image.
        Returns:
            Tensor: The output control points.
        """
        margin_x, margin_y = margins
        num_ctrl_pts_per_side = num_control_points // 2
        ctrl_pts_x = np.linspace(margin_x, 1.0 - margin_x,
                                 num_ctrl_pts_per_side)
        ctrl_pts_y_top = np.ones(num_ctrl_pts_per_side) * margin_y
        ctrl_pts_y_bottom = np.ones(num_ctrl_pts_per_side) * (1.0 - margin_y)
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        output_ctrl_pts_arr = np.concatenate([ctrl_pts_top, ctrl_pts_bottom],
                                             axis=0)
        output_ctrl_pts = torch.Tensor(output_ctrl_pts_arr)
        return output_ctrl_pts


@MODELS.register_module()
class STN(BasePreprocessor):
    """Implement STN module in ASTER: An Attentional Scene Text Recognizer with
    Flexible Rectification
    (https://ieeexplore.ieee.org/abstract/document/8395027/)

    Args:
        in_channels (int): The number of input channels.
        resized_image_size (Tuple[int, int]): The resized image size. The input
            image will be downsampled to have a better recitified result.
        output_image_size: The size of the output image for TPS. Defaults to
            (32, 100).
        num_control_points: The number of control points. Defaults to 20.
        margins: The margins for control points to the top and down side of the
            image for TPS. Defaults to [0.05, 0.05].
    """

    def __init__(self,
                 in_channels: int,
                 resized_image_size: Tuple[int, int] = (32, 64),
                 output_image_size: Tuple[int, int] = (32, 100),
                 num_control_points: int = 20,
                 margins: Tuple[float, float] = [0.05, 0.05],
                 init_cfg: Optional[Union[Dict, List[Dict]]] = [
                     dict(type='Xavier', layer='Conv2d'),
                     dict(type='Constant', val=1, layer='BatchNorm2d'),
                 ]):
        super().__init__(init_cfg=init_cfg)
        self.resized_image_size = resized_image_size
        self.num_control_points = num_control_points
        self.tps = TPStransform(output_image_size, num_control_points, margins)
        self.stn_convnet = nn.Sequential(
            ConvModule(in_channels, 32, 3, 1, 1, norm_cfg=dict(type='BN')),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvModule(32, 64, 3, 1, 1, norm_cfg=dict(type='BN')),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvModule(64, 128, 3, 1, 1, norm_cfg=dict(type='BN')),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvModule(128, 256, 3, 1, 1, norm_cfg=dict(type='BN')),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvModule(256, 256, 3, 1, 1, norm_cfg=dict(type='BN')),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvModule(256, 256, 3, 1, 1, norm_cfg=dict(type='BN')),
        )

        self.stn_fc1 = nn.Sequential(
            nn.Linear(2 * 256, 512), nn.BatchNorm1d(512),
            nn.ReLU(inplace=True))
        self.stn_fc2 = nn.Linear(512, num_control_points * 2)
        self.init_stn(self.stn_fc2)

    def init_stn(self, stn_fc2: nn.Linear) -> None:
        """Initialize the output linear layer of stn, so that the initial
        source point will be at the top and down side of the image, which will
        help to optimize.

        Args:
            stn_fc2 (nn.Linear): The output linear layer of stn.
        """
        margin = 0.01
        sampling_num_per_side = int(self.num_control_points / 2)
        ctrl_pts_x = np.linspace(margin, 1. - margin, sampling_num_per_side)
        ctrl_pts_y_top = np.ones(sampling_num_per_side) * margin
        ctrl_pts_y_bottom = np.ones(sampling_num_per_side) * (1 - margin)
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        ctrl_points = np.concatenate([ctrl_pts_top, ctrl_pts_bottom],
                                     axis=0).astype(np.float32)
        stn_fc2.weight.data.zero_()
        stn_fc2.bias.data = torch.Tensor(ctrl_points).view(-1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Forward function of STN.

        Args:
            img (Tensor): The input image tensor.

        Returns:
            Tensor: The rectified image tensor.
        """
        resize_img = F.interpolate(
            img, self.resized_image_size, mode='bilinear', align_corners=True)
        points = self.stn_convnet(resize_img)
        batch_size, _, _, _ = points.size()
        points = points.view(batch_size, -1)
        img_feat = self.stn_fc1(points)
        points = self.stn_fc2(0.1 * img_feat)
        points = points.view(-1, self.num_control_points, 2)

        transformd_image = self.tps(img, points)
        return transformd_image

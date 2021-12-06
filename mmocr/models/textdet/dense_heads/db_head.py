# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.runner import BaseModule, Sequential

from mmocr.models.builder import HEADS, build_loss
from .head_mixin import HeadMixin


@HEADS.register_module()
class DBHead(HeadMixin, BaseModule):
    """The class for DBNet head.

    This was partially adapted from https://github.com/MhLiao/DB

    Args:
        in_channels (int): The number of input channels of the db head.
        decoding_type (str): The type of decoder for dbnet.
        text_repr_type (str): Boundary encoding type 'poly' or 'quad'.
        downsample_ratio (float): The downsample ratio of ground truths.
        loss (dict): The type of loss for dbnet.
    """

    def __init__(self,
                 in_channels,
                 with_bias=False,
                 decoding_type='db',
                 text_repr_type='poly',
                 downsample_ratio=1.0,
                 loss=dict(type='DBLoss'),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=[
                     dict(type='Kaiming', layer='Conv'),
                     dict(
                         type='Constant', layer='BatchNorm', val=1., bias=1e-4)
                 ]):
        super().__init__(init_cfg=init_cfg)

        assert isinstance(in_channels, int)

        self.in_channels = in_channels
        self.text_repr_type = text_repr_type
        self.loss_module = build_loss(loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.downsample_ratio = downsample_ratio
        self.decoding_type = decoding_type

        self.binarize = Sequential(
            nn.Conv2d(
                in_channels, in_channels // 4, 3, bias=with_bias, padding=1),
            nn.BatchNorm2d(in_channels // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2),
            nn.BatchNorm2d(in_channels // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, 2, 2), nn.Sigmoid())

        self.threshold = self._init_thr(in_channels)

    def diff_binarize(self, prob_map, thr_map, k):
        return torch.reciprocal(1.0 + torch.exp(-k * (prob_map - thr_map)))

    def forward(self, inputs):
        """
        Args:
            inputs (Tensor): Shape (batch_size, hidden_size, h, w).

        Returns:
            Tensor: A tensor of the same shape as input.
        """
        prob_map = self.binarize(inputs)
        thr_map = self.threshold(inputs)
        binary_map = self.diff_binarize(prob_map, thr_map, k=50)
        outputs = torch.cat((prob_map, thr_map, binary_map), dim=1)
        return outputs

    def _init_thr(self, inner_channels, bias=False):
        in_channels = inner_channels
        seq = Sequential(
            nn.Conv2d(
                in_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.BatchNorm2d(inner_channels // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, inner_channels // 4, 2, 2),
            nn.BatchNorm2d(inner_channels // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, 1, 2, 2), nn.Sigmoid())
        return seq

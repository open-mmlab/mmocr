# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import BaseModule

from mmocr.models.builder import HEADS
from mmocr.utils import check_argument
from .base_head import BaseHead


@HEADS.register_module()
class PANHead(BaseHead, BaseModule):
    """The class for PANet head.

    Args:
        in_channels (list[int]): A list of 4 numbers of input channels.
        out_channels (int): Number of output channels.
        downsample_ratio (float): Downsample ratio.
        loss (dict): Configuration dictionary for loss type. Supported loss
            types are "PANLoss" and "PSELoss".
        postprocessor (dict): Config of postprocessor for PANet.
        train_cfg, test_cfg (dict): Depreciated.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 downsample_ratio=0.25,
                 loss=dict(type='PANLoss'),
                 postprocessor=dict(
                     type='PANPostprocessor', text_repr_type='poly'),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(
                     type='Normal',
                     mean=0,
                     std=0.01,
                     override=dict(name='out_conv')),
                 **kwargs):
        old_keys = ['text_repr_type', 'decoding_type']
        for key in old_keys:
            if kwargs.get(key, None):
                warnings.warn(
                    f'{key} is deprecated, please specify '
                    f'it in postprocessor config dict', UserWarning)

        BaseModule.__init__(self, init_cfg=init_cfg)
        BaseHead.__init__(self, loss, postprocessor)

        assert check_argument.is_type_list(in_channels, int)
        assert isinstance(out_channels, int)

        assert 0 <= downsample_ratio <= 1

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample_ratio = downsample_ratio
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.out_conv = nn.Conv2d(
            in_channels=np.sum(np.array(in_channels)),
            out_channels=out_channels,
            kernel_size=1)

    def forward(self, inputs):
        r"""
        Args:
            inputs (list[Tensor] | Tensor): Each tensor has the shape of
                :math:`(N, C_i, W, H)`, where :math:`\sum_iC_i=C_{in}` and
                :math:`C_{in}` is ``input_channels``.

        Returns:
            Tensor: A tensor of shape :math:`(N, C_{out}, W, H)` where
            :math:`C_{out}` is ``output_channels``.
        """
        if isinstance(inputs, tuple):
            outputs = torch.cat(inputs, dim=1)
        else:
            outputs = inputs
        outputs = self.out_conv(outputs)

        return outputs

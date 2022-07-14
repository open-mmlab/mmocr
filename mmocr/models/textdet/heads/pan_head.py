# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
import torch.nn as nn

from mmocr.data import TextDetDataSample
from mmocr.registry import MODELS
from mmocr.utils import check_argument
from .base_textdet_head import BaseTextDetHead


@MODELS.register_module()
class PANHead(BaseTextDetHead):
    """The class for PANet head.

    Args:
        in_channels (list[int]): A list of 4 numbers of input channels.
        hidden_dim (int): The hidden dimension of the first convolutional
            layer.
        out_channel (int): Number of output channels.
        module_loss (dict): Configuration dictionary for loss type. Defaults
            to dict(type='PANModuleLoss')
        postprocessor (dict): Config of postprocessor for PANet. Defaults to
            dict(type='PANPostprocessor', text_repr_type='poly').
        init_cfg (list[dict]): Initialization configs. Defaults to
            [dict(type='Normal', mean=0, std=0.01, layer='Conv2d'),
             dict(type='Constant', val=1, bias=0, layer='BN')]
    """

    def __init__(
        self,
        in_channels: List[int],
        hidden_dim: int,
        out_channel: int,
        module_loss=dict(type='PANModuleLoss'),
        postprocessor=dict(type='PANPostprocessor', text_repr_type='poly'),
        init_cfg=[
            dict(type='Normal', mean=0, std=0.01, layer='Conv2d'),
            dict(type='Constant', val=1, bias=0, layer='BN')
        ]
    ) -> None:
        super().__init__(
            module_loss=module_loss,
            postprocessor=postprocessor,
            init_cfg=init_cfg)

        assert check_argument.is_type_list(in_channels, int)
        assert isinstance(out_channel, int)
        assert isinstance(hidden_dim, int)

        in_channels = sum(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            hidden_dim, out_channel, kernel_size=1, stride=1, padding=0)

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[TextDetDataSample]] = None
                ) -> torch.Tensor:
        r"""PAN head forward.
        Args:
            inputs (list[Tensor] | Tensor): Each tensor has the shape of
                :math:`(N, C_i, W, H)`, where :math:`\sum_iC_i=C_{in}` and
                :math:`C_{in}` is ``input_channels``.
            data_samples (list[TextDetDataSample], optional): A list of data
                samples. Defaults to None.

        Returns:
            Tensor: A tensor of shape :math:`(N, C_{out}, W, H)` where
            :math:`C_{out}` is ``output_channels``.
        """
        if isinstance(inputs, tuple):
            outputs = torch.cat(inputs, dim=1)
        else:
            outputs = inputs
        outputs = self.conv1(outputs)
        outputs = self.relu1(self.bn1(outputs))
        outputs = self.conv2(outputs)
        return outputs

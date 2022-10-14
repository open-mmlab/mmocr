# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

import torch
import torch.nn as nn

from mmocr.registry import MODELS


@MODELS.register_module()
class AvgPool2d(nn.Module):
    """Applies a 2D average pooling over an input signal composed of several
    input planes.

    AvgPool2d class for plug-in manner usage

    Args:
        kernel_size (int | tuple(int)): the size of the window.
        stride (int | tuple(int)): the stride of the window.
        padding (int | tuple(int)): implicit zero padding.
    """

    def __init__(self,
                 kernel_size: Union[int, Tuple[int]],
                 stride: Union[int, Tuple[int]] = None,
                 padding: Union[int, Tuple[int]] = 0,
                 **kwargs) -> None:
        super().__init__()
        self.model = nn.AvgPool2d(kernel_size, stride, padding)

    def forward(self, x) -> torch.Tensor:
        """Forward function.
        Args:
            x (Tensor): Input feature map.

        Returns:
            Tensor: Output tensor after Maxpooling layer.
        """
        return self.model(x)

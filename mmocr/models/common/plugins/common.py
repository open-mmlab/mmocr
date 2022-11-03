# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from mmocr.registry import MODELS


@MODELS.register_module()
class AvgPool2d(nn.Module):
    """Applies a 2D average pooling over an input signal composed of several
    input planes.

    It can also be used as a network plugin.

    Args:
        kernel_size (int or tuple(int)): the size of the window.
        stride (int or tuple(int), optional): the stride of the window.
            Defaults to None.
        padding (int or tuple(int)): implicit zero padding. Defaults to 0.
    """

    def __init__(self,
                 kernel_size: Union[int, Tuple[int]],
                 stride: Optional[Union[int, Tuple[int]]] = None,
                 padding: Union[int, Tuple[int]] = 0,
                 **kwargs) -> None:
        super().__init__()
        self.model = nn.AvgPool2d(kernel_size, stride, padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.
        Args:
            x (Tensor): Input feature map.

        Returns:
            Tensor: Output tensor after Avgpooling layer.
        """
        return self.model(x)

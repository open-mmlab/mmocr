# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import PLUGIN_LAYERS


@PLUGIN_LAYERS.register_module()
class Maxpool2d(nn.Module):
    """nn.Maxpool2d()
    Args:
        kernel_size (int | tuple): Kernel size for max pooling layer
        stride (int | tuple): Stride for max pooling layer
        padding (int | tuple): Padding for pooling layer
    """

    def __init__(self, kernel_size, stride, padding=0, **kwargs):
        super(Maxpool2d, self).__init__()
        self.model = nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input feature map

        Returns:
            Tensor: The tensor after Maxpooling layer.
        """
        return self.model(x)

# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmengine.model import BaseModule, Sequential

from mmocr.registry import MODELS


@MODELS.register_module()
class VeryDeepVgg(BaseModule):
    """Implement VGG-VeryDeep backbone for text recognition, modified from
    `VGG-VeryDeep <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        leaky_relu (bool): Use leakyRelu or not.
        input_channels (int): Number of channels of input image tensor.
    """

    def __init__(self,
                 leaky_relu=True,
                 input_channels=3,
                 init_cfg=[
                     dict(type='Xavier', layer='Conv2d'),
                     dict(type='Uniform', layer='BatchNorm2d')
                 ]):
        super().__init__(init_cfg=init_cfg)

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        self.channels = nm

        # cnn = nn.Sequential()
        cnn = Sequential()

        def conv_relu(i, batch_normalization=False):
            n_in = input_channels if i == 0 else nm[i - 1]
            n_out = nm[i]
            cnn.add_module(f'conv{i}',
                           nn.Conv2d(n_in, n_out, ks[i], ss[i], ps[i]))
            if batch_normalization:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(n_out))
            if leaky_relu:
                cnn.add_module(f'relu{i}', nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module(f'relu{i}', nn.ReLU(True))

        conv_relu(0)
        cnn.add_module(f'pooling{0}', nn.MaxPool2d(2, 2))  # 64x16x64
        conv_relu(1)
        cnn.add_module(f'pooling{1}', nn.MaxPool2d(2, 2))  # 128x8x32
        conv_relu(2, True)
        conv_relu(3)
        cnn.add_module(f'pooling{2}', nn.MaxPool2d((2, 2), (2, 1),
                                                   (0, 1)))  # 256x4x16
        conv_relu(4, True)
        conv_relu(5)
        cnn.add_module(f'pooling{3}', nn.MaxPool2d((2, 2), (2, 1),
                                                   (0, 1)))  # 512x2x16
        conv_relu(6, True)  # 512x1x16

        self.cnn = cnn

    def out_channels(self):
        return self.channels[-1]

    def forward(self, x):
        """
        Args:
            x (Tensor): Images of shape :math:`(N, C, H, W)`.

        Returns:
            Tensor: The feature Tensor of shape :math:`(N, 512, H/32, (W/4+1)`.
        """
        output = self.cnn(x)

        return output

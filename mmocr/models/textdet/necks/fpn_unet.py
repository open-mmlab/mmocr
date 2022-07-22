# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from mmcv.runner import BaseModule
from torch import nn

from mmocr.registry import MODELS


class UpBlock(BaseModule):
    """Upsample block for DRRG and TextSnake.

    DRRG: `Deep Relational Reasoning Graph Network for Arbitrary Shape
    Text Detection <https://arxiv.org/abs/2003.07493>`_.

    TextSnake: `A Flexible Representation for Detecting Text of Arbitrary
    Shapes <https://arxiv.org/abs/1807.01544>`_.

    Args:
        in_channels (list[int]): Number of input channels at each scale. The
            length of the list should be 4.
        out_channels (int): The number of output channels.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None) -> None:
        super().__init__(init_cfg=init_cfg)

        assert isinstance(in_channels, int)
        assert isinstance(out_channels, int)

        self.conv1x1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation."""
        x = F.relu(self.conv1x1(x))
        x = F.relu(self.conv3x3(x))
        x = self.deconv(x)
        return x


@MODELS.register_module()
class FPN_UNet(BaseModule):
    """The class for implementing DRRG and TextSnake U-Net-like FPN.

    DRRG: `Deep Relational Reasoning Graph Network for Arbitrary Shape
    Text Detection <https://arxiv.org/abs/2003.07493>`_.

    TextSnake: `A Flexible Representation for Detecting Text of Arbitrary
    Shapes <https://arxiv.org/abs/1807.01544>`_.

    Args:
        in_channels (list[int]): Number of input channels at each scale. The
            length of the list should be 4.
        out_channels (int): The number of output channels.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        init_cfg: Optional[Union[Dict, List[Dict]]] = dict(
            type='Xavier',
            layer=['Conv2d', 'ConvTranspose2d'],
            distribution='uniform')
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        assert len(in_channels) == 4
        assert isinstance(out_channels, int)

        blocks_out_channels = [out_channels] + [
            min(out_channels * 2**i, 256) for i in range(4)
        ]
        blocks_in_channels = [blocks_out_channels[1]] + [
            in_channels[i] + blocks_out_channels[i + 2] for i in range(3)
        ] + [in_channels[3]]

        self.up4 = nn.ConvTranspose2d(
            blocks_in_channels[4],
            blocks_out_channels[4],
            kernel_size=4,
            stride=2,
            padding=1)
        self.up_block3 = UpBlock(blocks_in_channels[3], blocks_out_channels[3])
        self.up_block2 = UpBlock(blocks_in_channels[2], blocks_out_channels[2])
        self.up_block1 = UpBlock(blocks_in_channels[1], blocks_out_channels[1])
        self.up_block0 = UpBlock(blocks_in_channels[0], blocks_out_channels[0])

    def forward(self, x: List[Union[torch.Tensor,
                                    Tuple[torch.Tensor]]]) -> torch.Tensor:
        """
        Args:
            x (list[Tensor] | tuple[Tensor]): A list of four tensors of shape
                :math:`(N, C_i, H_i, W_i)`, representing C2, C3, C4, C5
                features respectively. :math:`C_i` should matches the number in
                ``in_channels``.

        Returns:
            Tensor: Shape :math:`(N, C, H, W)` where :math:`H=4H_0` and
            :math:`W=4W_0`.
        """
        c2, c3, c4, c5 = x

        x = F.relu(self.up4(c5))

        c4 = F.interpolate(
            c4, size=x.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, c4], dim=1)
        x = F.relu(self.up_block3(x))

        c3 = F.interpolate(
            c3, size=x.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, c3], dim=1)
        x = F.relu(self.up_block2(x))

        c2 = F.interpolate(
            c2, size=x.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, c2], dim=1)
        x = F.relu(self.up_block1(x))

        x = self.up_block0(x)
        # the output should be of the same height and width as backbone input
        return x

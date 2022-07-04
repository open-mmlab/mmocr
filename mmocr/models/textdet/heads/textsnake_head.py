# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from mmocr.core import TextDetDataSample
from mmocr.models.textdet.heads import BaseTextDetHead
from mmocr.registry import MODELS


@MODELS.register_module()
class TextSnakeHead(BaseTextDetHead):
    """The class for TextSnake head: TextSnake: A Flexible Representation for
    Detecting Text of Arbitrary Shapes.

    TextSnake: `A Flexible Representation for Detecting Text of Arbitrary
    Shapes <https://arxiv.org/abs/1807.01544>`_.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        downsample_ratio (float): Downsample ratio.
        loss (dict): Configuration dictionary for loss type.
        postprocessor (dict): Config of postprocessor for TextSnake.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 5,
        downsample_ratio: float = 1.0,
        loss_module: Dict = dict(type='TextSnakeLoss'),
        postprocessor: Dict = dict(
            type='TextSnakePostprocessor', text_repr_type='poly'),
        init_cfg: Optional[Union[Dict, List[Dict]]] = dict(
            type='Normal', override=dict(name='out_conv'), mean=0, std=0.01)
    ) -> None:
        super().__init__(
            loss_module=loss_module,
            postprocessor=postprocessor,
            init_cfg=init_cfg)
        assert isinstance(in_channels, int)
        assert isinstance(out_channels, int)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample_ratio = downsample_ratio

        self.out_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0)

    def forward(self, inputs: torch.Tensor,
                data_samples: List[TextDetDataSample]) -> Dict:
        """
        Args:
            inputs (torch.Tensor): Shape :math:`(N, C_{in}, H, W)`, where
                :math:`C_{in}` is ``in_channels``. :math:`H` and :math:`W`
                should be the same as the input of backbone.
            data_samples (List[TextDetDataSample]): List of data samples.

        Returns:
            Tensor: A tensor of shape :math:`(N, 5, H, W)`, where the five
            channels represent [0]: text score, [1]: center score,
            [2]: sin, [3] cos, [4] radius, respectively.
        """
        outputs = self.out_conv(inputs)
        return outputs

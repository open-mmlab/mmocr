# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from mmocr.data import TextDetDataSample
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
        module_loss (dict): Configuration dictionary for loss type.
            Defaults to ``dict(type='TextSnakeModuleLoss')``.
        postprocessor (dict): Config of postprocessor for TextSnake.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 5,
        downsample_ratio: float = 1.0,
        module_loss: Dict = dict(type='TextSnakeModuleLoss'),
        postprocessor: Dict = dict(
            type='TextSnakePostprocessor', text_repr_type='poly'),
        init_cfg: Optional[Union[Dict, List[Dict]]] = dict(
            type='Normal', override=dict(name='out_conv'), mean=0, std=0.01)
    ) -> None:
        super().__init__(
            module_loss=module_loss,
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

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[TextDetDataSample]] = None
                ) -> Dict:
        """
        Args:
            inputs (torch.Tensor): Shape :math:`(N, C_{in}, H, W)`, where
                :math:`C_{in}` is ``in_channels``. :math:`H` and :math:`W`
                should be the same as the input of backbone.
            data_samples (list[TextDetDataSample], optional): A list of data
                samples. Defaults to None.

        Returns:
            Tensor: A tensor of shape :math:`(N, 5, H, W)`, where the five
            channels represent [0]: text score, [1]: center score,
            [2]: sin, [3] cos, [4] radius, respectively.
        """
        outputs = self.out_conv(inputs)
        return outputs

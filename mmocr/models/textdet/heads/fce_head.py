# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from mmdet.core import multi_apply

from mmocr.core import TextDetDataSample
from mmocr.models.textdet.heads import BaseTextDetHead
from mmocr.registry import MODELS


@MODELS.register_module()
class FCEHead(BaseTextDetHead):
    """The class for implementing FCENet head.

    FCENet(CVPR2021): `Fourier Contour Embedding for Arbitrary-shaped Text
    Detection <https://arxiv.org/abs/2104.10442>`_

    Args:
        in_channels (int): The number of input channels.
        fourier_degree (int) : The maximum Fourier transform degree k. Defaults
            to 5.
        loss (dict): Config of loss for FCENet.
        postprocessor (dict): Config of postprocessor for FCENet.
        init_cfg (dict, optional): Initialization configs.
    """

    def __init__(
        self,
        in_channels: int,
        fourier_degree: int = 5,
        loss_module: Dict = dict(type='FCELoss', num_sample=50),
        postprocessor: Dict = dict(
            type='FCEPostprocessor',
            text_repr_type='poly',
            num_reconstr_points=50,
            alpha=1.0,
            beta=2.0,
            score_thr=0.3),
        init_cfg: Optional[Dict] = dict(
            type='Normal',
            mean=0,
            std=0.01,
            override=[dict(name='out_conv_cls'),
                      dict(name='out_conv_reg')])
    ) -> None:
        loss_module['fourier_degree'] = fourier_degree
        postprocessor['fourier_degree'] = fourier_degree
        super().__init__(
            loss_module=loss_module,
            postprocessor=postprocessor,
            init_cfg=init_cfg)

        assert isinstance(in_channels, int)
        assert isinstance(fourier_degree, int)

        self.in_channels = in_channels
        self.fourier_degree = fourier_degree
        self.out_channels_cls = 4
        self.out_channels_reg = (2 * self.fourier_degree + 1) * 2

        self.out_conv_cls = nn.Conv2d(
            self.in_channels,
            self.out_channels_cls,
            kernel_size=3,
            stride=1,
            padding=1)
        self.out_conv_reg = nn.Conv2d(
            self.in_channels,
            self.out_channels_reg,
            kernel_size=3,
            stride=1,
            padding=1)

    def forward(self,
                inputs: List[torch.Tensor],
                data_samples: Optional[List[TextDetDataSample]] = None
                ) -> Dict:
        """
        Args:
            inputs (List[Tensor]): Each tensor has the shape of :math:`(N, C_i,
                H_i, W_i)`.
            data_samples (list[TextDetDataSample], optional): A list of data
                samples. Defaults to None.

        Returns:
            list[dict]: A list of dict with keys of ``cls_res``, ``reg_res``
            corresponds to the classification result and regression result
            computed from the input tensor with the same index. They have
            the shapes of :math:`(N, C_{cls,i}, H_i, W_i)` and :math:`(N,
            C_{out,i}, H_i, W_i)`.
        """
        cls_res, reg_res = multi_apply(self.forward_single, inputs)
        level_num = len(cls_res)
        preds = [
            dict(cls_res=cls_res[i], reg_res=reg_res[i])
            for i in range(level_num)
        ]
        return preds

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for a single feature level.

        Args:
            x (Tensor): The input tensor with the shape of :math:`(N, C_i,
                H_i, W_i)`.

        Returns:
            Tensor: The classification and regression result with the shape of
            :math:`(N, C_{cls,i}, H_i, W_i)` and :math:`(N, C_{out,i}, H_i,
            W_i)`.
        """
        cls_predict = self.out_conv_cls(x)
        reg_predict = self.out_conv_reg(x)
        return cls_predict, reg_predict

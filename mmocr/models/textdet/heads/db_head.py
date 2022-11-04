# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.model import Sequential
from torch import Tensor

from mmocr.models.textdet.heads import BaseTextDetHead
from mmengine.registry import MODELS
from mmocr.structures import TextDetDataSample
from mmocr.utils.typing import DetSampleList


@MODELS.register_module()
class DBHead(BaseTextDetHead):
    """The class for DBNet head.

    This was partially adapted from https://github.com/MhLiao/DB

    Args:
        in_channels (int): The number of input channels.
        with_bias (bool): Whether add bias in Conv2d layer. Defaults to False.
        module_loss (dict): Config of loss for dbnet. Defaults to
            ``dict(type='DBModuleLoss')``
        postprocessor (dict): Config of postprocessor for dbnet.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(
        self,
        in_channels: int,
        with_bias: bool = False,
        module_loss: Dict = dict(type='DBModuleLoss'),
        postprocessor: Dict = dict(
            type='DBPostprocessor', text_repr_type='quad'),
        init_cfg: Optional[Union[Dict, List[Dict]]] = [
            dict(type='Kaiming', layer='Conv'),
            dict(type='Constant', layer='BatchNorm', val=1., bias=1e-4)
        ]
    ) -> None:
        super().__init__(
            module_loss=module_loss,
            postprocessor=postprocessor,
            init_cfg=init_cfg)

        assert isinstance(in_channels, int)
        assert isinstance(with_bias, bool)

        self.in_channels = in_channels
        self.binarize = Sequential(
            nn.Conv2d(
                in_channels, in_channels // 4, 3, bias=with_bias, padding=1),
            nn.BatchNorm2d(in_channels // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2),
            nn.BatchNorm2d(in_channels // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, 2, 2))
        self.threshold = self._init_thr(in_channels)
        self.sigmoid = nn.Sigmoid()

    def _diff_binarize(self, prob_map: Tensor, thr_map: Tensor,
                       k: int) -> Tensor:
        """Differential binarization.

        Args:
            prob_map (Tensor): Probability map.
            thr_map (Tensor): Threshold map.
            k (int): Amplification factor.

        Returns:
            Tensor: Binary map.
        """
        return torch.reciprocal(1.0 + torch.exp(-k * (prob_map - thr_map)))

    def _init_thr(self,
                  inner_channels: int,
                  bias: bool = False) -> nn.ModuleList:
        """Initialize threshold branch."""
        in_channels = inner_channels
        seq = Sequential(
            nn.Conv2d(
                in_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.BatchNorm2d(inner_channels // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, inner_channels // 4, 2, 2),
            nn.BatchNorm2d(inner_channels // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, 1, 2, 2), nn.Sigmoid())
        return seq

    def forward(self,
                img: Tensor,
                data_samples: Optional[List[TextDetDataSample]] = None,
                mode: str = 'predict') -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            img (Tensor): Shape :math:`(N, C, H, W)`.
            data_samples (list[TextDetDataSample], optional): A list of data
                samples. Defaults to None.
            mode (str): Forward mode. It affects the return values. Options are
                "loss", "predict" and "both". Defaults to "predict".

                - ``loss``: Run the full network and return the prob
                  logits, threshold map and binary map.
                - ``predict``: Run the binarzation part and return the prob
                  map only.
                - ``both``: Run the full network and return prob logits,
                  threshold map, binary map and prob map.

        Returns:
            Tensor or tuple(Tensor): Its type depends on ``mode``, read its
            docstring for details. Each has the shape of
            :math:`(N, 4H, 4W)`.
        """
        prob_logits = self.binarize(img).squeeze(1)
        prob_map = self.sigmoid(prob_logits)
        if mode == 'predict':
            return prob_map
        thr_map = self.threshold(img).squeeze(1)
        binary_map = self._diff_binarize(prob_map, thr_map, k=50).squeeze(1)
        if mode == 'loss':
            return prob_logits, thr_map, binary_map
        return prob_logits, thr_map, binary_map, prob_map

    def loss(self, x: Tuple[Tensor],
             batch_data_samples: DetSampleList) -> Dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        outs = self(x, batch_data_samples, mode='loss')
        losses = self.module_loss(outs, batch_data_samples)
        return losses

    def loss_and_predict(self, x: Tuple[Tensor],
                         batch_data_samples: DetSampleList
                         ) -> Tuple[dict, DetSampleList]:
        """Perform forward propagation of the head, then calculate loss and
        predictions from the features and data samples.

        Args:
            x (tuple[Tensor]): Features from FPN.
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple: the return value is a tuple contains:

                - losses: (dict[str, Tensor]): A dictionary of loss components.
                - predictions (list[:obj:`InstanceData`]): Detection
                  results of each image after the post process.
        """
        outs = self(x, batch_data_samples, mode='both')
        losses = self.module_loss(outs[:3], batch_data_samples)
        predictions = self.postprocessor(outs[3], batch_data_samples)
        return losses, predictions

    def predict(self, x: torch.Tensor,
                batch_data_samples: DetSampleList) -> DetSampleList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            SampleList: Detection results of each image
            after the post process.
        """
        outs = self(x, batch_data_samples, mode='predict')
        predictions = self.postprocessor(outs, batch_data_samples)
        return predictions

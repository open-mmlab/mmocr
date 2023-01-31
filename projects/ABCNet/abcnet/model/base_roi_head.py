# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Tuple

from torch import Tensor

from mmengine.model import BaseModule
from mmocr.utils import DetSampleList


class BaseRoIHead(BaseModule, metaclass=ABCMeta):
    """Base class for RoIHeads."""

    @property
    def with_rec_head(self):
        """bool: whether the RoI head contains a `mask_head`"""
        return hasattr(self, 'rec_head') and self.rec_head is not None

    @property
    def with_extractor(self):
        """bool: whether the RoI head contains a `mask_head`"""
        return hasattr(self,
                       'roi_extractor') and self.roi_extractor is not None

    # @abstractmethod
    # def init_assigner_sampler(self, *args, **kwargs):
    #     """Initialize assigner and sampler."""
    #     pass

    @abstractmethod
    def loss(self, x: Tuple[Tensor], data_samples: DetSampleList):
        """Perform forward propagation and loss calculation of the roi head on
        the features of the upstream network."""

    @abstractmethod
    def predict(self, x: Tuple[Tensor],
                data_samples: DetSampleList) -> DetSampleList:
        """Perform forward propagation of the roi head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (N, C, H, W).
            data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes `gt_instance`

        Returns:
            list[obj:`DetDataSample`]: Detection results of each image.
            Each item usually contains following keys in 'pred_instance'

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - polygon (List[Tensor]): Has a shape (num_instances, H, W).
        """

# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Dict, Tuple, Union

import torch
from mmengine.model import BaseModel
from torch import Tensor

from mmocr.utils.typing import (DetSampleList, OptConfigType, OptDetSampleList,
                                OptMultiConfig)

ForwardResults = Union[Dict[str, torch.Tensor], DetSampleList,
                       Tuple[torch.Tensor], torch.Tensor]


class BaseTextDetector(BaseModel, metaclass=ABCMeta):
    """Base class for detectors.

    Args:
       data_preprocessor (dict or ConfigDict, optional): The pre-process
           config of :class:`BaseDataPreprocessor`.  it usually includes,
            ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
       init_cfg (dict or ConfigDict, optional): the config to control the
           initialization. Defaults to None.
    """

    def __init__(self,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

    @property
    def with_neck(self) -> bool:
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck') and self.neck is not None

    def forward(self,
                batch_inputs: torch.Tensor,
                batch_data_samples: OptDetSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`TextDetDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            batch_inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            batch_data_samples (list[:obj:`TextDetDataSample`], optional): The
                annotation data of every samples. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`TextDetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(batch_inputs, batch_data_samples)
        elif mode == 'predict':
            return self.predict(batch_inputs, batch_data_samples)
        elif mode == 'tensor':
            return self._forward(batch_inputs, batch_data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    @abstractmethod
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: DetSampleList) -> Union[dict, tuple]:
        """Calculate losses from a batch of inputs and data samples."""
        pass

    @abstractmethod
    def predict(self, batch_inputs: Tensor,
                batch_data_samples: DetSampleList) -> DetSampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing."""
        pass

    @abstractmethod
    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptDetSampleList = None):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass

    @abstractmethod
    def extract_feat(self, batch_inputs: Tensor):
        """Extract features from images."""
        pass

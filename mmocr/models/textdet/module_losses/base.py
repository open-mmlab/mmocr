# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Dict, Sequence, Tuple, Union

import torch
from torch import nn

from mmengine.registry import MODELS
from mmocr.utils.typing import DetSampleList

INPUT_TYPES = Union[torch.Tensor, Sequence[torch.Tensor], Dict]


@MODELS.register_module()
class BaseTextDetModuleLoss(nn.Module, metaclass=ABCMeta):
    r"""Base class for text detection module loss.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self,
                inputs: INPUT_TYPES,
                data_samples: DetSampleList = None) -> Dict:
        """Calculates losses from a batch of inputs and data samples. Returns a
        dict of losses.

        Args:
            inputs (Tensor or list[Tensor] or dict): The raw tensor outputs
                from the model.
            data_samples (list(TextDetDataSample)): Datasamples containing
                ground truth data.

        Returns:
            dict: A dict of losses.
        """
        pass

    @abstractmethod
    def get_targets(self, data_samples: DetSampleList) -> Tuple:
        """Generates loss targets from data samples. Returns a tuple of target
        tensors.

        Args:
            data_samples (list(TextDetDataSample)): Ground truth data samples.

        Returns:
            tuple: A tuple of target tensors.
        """
        pass

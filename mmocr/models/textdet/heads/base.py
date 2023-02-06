# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
from mmengine.model import BaseModule
from torch import Tensor

from mmocr.registry import MODELS
from mmocr.utils.typing_utils import DetSampleList


@MODELS.register_module()
class BaseTextDetHead(BaseModule):
    """Base head for text detection, build the loss and postprocessor.

    1. The ``init_weights`` method is used to initialize head's
    model parameters. After detector initialization, ``init_weights``
    is triggered when ``detector.init_weights()`` is called externally.

    2. The ``loss`` method is used to calculate the loss of head,
    which includes two steps: (1) the head model performs forward
    propagation to obtain the feature maps (2) The ``module_loss`` method
    is called based on the feature maps to calculate the loss.

    .. code:: text

        loss(): forward() -> module_loss()

    3. The ``predict`` method is used to predict detection results,
    which includes two steps: (1) the head model performs forward
    propagation to obtain the feature maps (2) The ``postprocessor`` method
    is called based on the feature maps to predict detection results including
    post-processing.

    .. code:: text

        predict(): forward() -> postprocessor()

    4. The ``loss_and_predict`` method is used to return loss and detection
    results at the same time. It will call head's ``forward``,
    ``module_loss`` and ``postprocessor`` methods in order.

    .. code:: text

        loss_and_predict(): forward() -> module_loss() -> postprocessor()


    Args:
        loss (dict, optional): Config to build loss. Defaults to None.
        postprocessor (dict, optional): Config to build postprocessor. Defaults
            to None.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """

    def __init__(self,
                 module_loss: Optional[Dict] = None,
                 postprocessor: Optional[Dict] = None,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        if module_loss is not None:
            assert isinstance(module_loss, dict)
            self.module_loss = MODELS.build(module_loss)
        else:
            self.module_loss = module_loss
        if postprocessor is not None:
            assert isinstance(postprocessor, dict)
            self.postprocessor = MODELS.build(postprocessor)
        else:
            self.postprocessor = postprocessor

    def loss(self, x: Tuple[Tensor], data_samples: DetSampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        outs = self(x, data_samples)
        losses = self.module_loss(outs, data_samples)
        return losses

    def loss_and_predict(self, x: Tuple[Tensor], data_samples: DetSampleList
                         ) -> Tuple[dict, DetSampleList]:
        """Perform forward propagation of the head, then calculate loss and
        predictions from the features and data samples.

        Args:
            x (tuple[Tensor]): Features from FPN.
            data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple: the return value is a tuple contains:

                - losses: (dict[str, Tensor]): A dictionary of loss components.
                - predictions (list[:obj:`InstanceData`]): Detection
                  results of each image after the post process.
        """
        outs = self(x, data_samples)
        losses = self.module_loss(outs, data_samples)

        predictions = self.postprocessor(outs, data_samples, self.training)
        return losses, predictions

    def predict(self, x: torch.Tensor,
                data_samples: DetSampleList) -> DetSampleList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            SampleList: Detection results of each image
            after the post process.
        """
        outs = self(x, data_samples)

        predictions = self.postprocessor(outs, data_samples)
        return predictions

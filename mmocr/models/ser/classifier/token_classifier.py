# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple, Union

import torch
from mmengine.model import BaseModel

from mmocr.registry import MODELS
from mmocr.utils.typing_utils import OptSERSampleList, SERSampleList

ForwardResults = Union[Dict[str, torch.Tensor], SERSampleList,
                       Tuple[torch.Tensor], torch.Tensor]


@MODELS.register_module()
class LayoutLMv3TokenClassifier(BaseModel):

    def __init__(self,
                 backbone: Dict,
                 cls_head: Dict,
                 data_preprocessor: Optional[Dict] = None,
                 init_cfg: Optional[Dict] = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        assert cls_head is not None, 'cls_head cannot be None!'
        # self.backbone = MODELS.build(backbone)
        # self.cls_head = MODELS.build(cls_head)

    def forward(self,
                inputs: torch.Tensor,
                data_samples: OptSERSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`SERDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle either back propagation or
        parameter update, which are supposed to be done in :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (list[:obj:`SERDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`SERDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        data = self.data_preprocessor(inputs, True)
        print(data)
        # if mode == 'loss':
        #     return self.loss(inputs, data_samples)
        # elif mode == 'predict':
        #     return self.predict(inputs, data_samples)
        # elif mode == 'tensor':
        #     return self._forward(inputs, data_samples)
        # else:
        #     raise RuntimeError(f'Invalid mode "{mode}". '
        #                        'Only supports loss, predict and tensor mode')

# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple, Union

import torch
from mmengine.model import BaseModel

from mmocr.registry import MODELS
from projects.LayoutLMv3.utils.typing_utils import (OptSERSampleList,
                                                    SERSampleList)
from transformers import LayoutLMv3ForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput

ForwardResults = Union[Dict[str, torch.Tensor], SERSampleList,
                       Tuple[torch.Tensor], torch.Tensor]


@MODELS.register_module()
class HFLayoutLMv3ForTokenClassificationWrapper(BaseModel):

    def __init__(self,
                 layoutlmv3_token_classifier: dict = dict(
                     pretrained_model_name_or_path=None),
                 loss_processor: Optional[Dict] = None,
                 data_preprocessor: Optional[Dict] = None,
                 postprocessor: Optional[Dict] = None,
                 init_cfg: Optional[Dict] = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if isinstance(layoutlmv3_token_classifier, dict) and \
                layoutlmv3_token_classifier.get(
                    'pretrained_model_name_or_path', None):
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(
                **layoutlmv3_token_classifier)
        else:
            raise TypeError(
                'layoutlmv3_token_classifier cfg should be a `dict` and a key '
                '`pretrained_model_name_or_path` must be specified')

        if loss_processor is not None:
            assert isinstance(loss_processor, dict)
            self.loss_processor = MODELS.build(loss_processor)

        if postprocessor is not None:
            assert isinstance(postprocessor, dict)
            self.postprocessor = MODELS.build(postprocessor)

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
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def loss(self, inputs: torch.Tensor, data_samples: SERSampleList) -> Dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (torch.Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            data_samples (list[SERDataSample]): A list of N
                datasamples, containing meta information and gold annotations
                for each of the images.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        labels = inputs.pop('labels')
        outputs: TokenClassifierOutput = self.model(**inputs)
        return self.loss_processor(outputs, labels)

    def predict(self, inputs: torch.Tensor,
                data_samples: SERSampleList) -> SERSampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (torch.Tensor): Images of shape (N, C, H, W).
            data_samples (list[SERDataSample]): A list of N
                datasamples, containing meta information and gold annotations
                for each of the images.

        Returns:
            list[SERDataSample]: A list of N datasamples of prediction
            results.  Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - polygons (list[np.ndarray]): The length is num_instances.
                    Each element represents the polygon of the
                    instance, in (xn, yn) order.
        """
        outputs: TokenClassifierOutput = self.model(**inputs)
        return self.postprocessor(outputs['logits'], data_samples)

    def _forward(self,
                 inputs: torch.Tensor,
                 data_samples: OptSERSampleList = None,
                 **kwargs) -> torch.Tensor:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (list[SERDataSample]): A list of N
                datasamples, containing meta information and gold annotations
                for each of the images.

        Returns:
            Tensor or tuple[Tensor]: A tuple of features from ``det_head``
            forward.
        """
        return self.model(**inputs)

# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from mmdet.data_elements.bbox import bbox2roi
from mmengine.model import BaseModel
from torch import nn

from mmocr.data import KIEDataSample
from mmocr.registry import MODELS, TASK_UTILS


@MODELS.register_module()
class SDMGR(BaseModel):
    """The implementation of the paper: Spatial Dual-Modality Graph Reasoning
    for Key Information Extraction. https://arxiv.org/abs/2103.14470.

    Args:
        backbone (dict, optional): Config of backbone. If None, None will be
            passed to kie_head during training and testing. Defaults to None.
        roi_extractor (dict, optional): Config of roi extractor. Only
            applicable when backbone is not None. Defaults to None.
        neck (dict, optional): Config of neck. Defaults to None.
        kie_head (dict): Config of KIE head. Defaults to None.
        dictionary (dict, optional): Config of dictionary. Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
            ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``. It has
            to be None when working in non-visual mode. Defaults to None.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """

    def __init__(self,
                 backbone: Optional[Dict] = None,
                 roi_extractor: Optional[Dict] = None,
                 neck: Optional[Dict] = None,
                 kie_head: Dict = None,
                 dictionary: Optional[Dict] = None,
                 data_preprocessor: Optional[Dict] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if dictionary is not None:
            self.dictionary = TASK_UTILS.build(dictionary)
            if kie_head.get('dictionary', None) is None:
                kie_head.update(dictionary=self.dictionary)
            else:
                warnings.warn(f"Using dictionary {kie_head['dictionary']} "
                              "in kie_head's config.")
        if backbone is not None:
            self.backbone = MODELS.build(backbone)
            self.extractor = MODELS.build({
                **roi_extractor, 'out_channels':
                self.backbone.base_channels
            })
            self.maxpool = nn.MaxPool2d(
                roi_extractor['roi_layer']['output_size'])
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.kie_head = MODELS.build(kie_head)

    def extract_feat(self, img: torch.Tensor,
                     gt_bboxes: List[torch.Tensor]) -> torch.Tensor:
        """Extract features from images if self.backbone is not None. It
        returns None otherwise.

        Args:
            img (torch.Tensor): The input image with shape (N, C, H, W).
            gt_bboxes (list[torch.Tensor)): A list of ground truth bounding
                boxes, each of shape :math:`(N_i, 4)`.

        Returns:
            torch.Tensor: The extracted features with shape (N, E).
        """
        if not hasattr(self, 'backbone'):
            return None
        x = self.backbone(img)
        if hasattr(self, 'neck'):
            x = self.neck(x)
        x = x[-1]
        feats = self.maxpool(self.extractor([x], bbox2roi(gt_bboxes)))
        return feats.view(feats.size(0), -1)

    def forward(self,
                batch_inputs: torch.Tensor,
                batch_data_samples: Sequence[KIEDataSample] = None,
                mode: str = 'tensor',
                **kwargs) -> torch.Tensor:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`DetDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            batch_inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            batch_data_samples (list[:obj:`DetDataSample`], optional): The
                annotation data of every samples. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(batch_inputs, batch_data_samples, **kwargs)
        elif mode == 'predict':
            return self.predict(batch_inputs, batch_data_samples, **kwargs)
        elif mode == 'tensor':
            return self._forward(batch_inputs, batch_data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def loss(self, batch_inputs: torch.Tensor,
             batch_data_samples: Sequence[KIEDataSample], **kwargs) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (torch.Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            batch_data_samples (list[KIEDataSample]): A list of N datasamples,
                containing meta information and gold annotations for each of
                the images.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs, [
            data_sample.gt_instances.bboxes
            for data_sample in batch_data_samples
        ])
        return self.kie_head.loss(x, batch_data_samples)

    def predict(self, batch_inputs: torch.Tensor,
                batch_data_samples: Sequence[KIEDataSample],
                **kwargs) -> List[KIEDataSample]:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        Args:
            batch_inputs (torch.Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            batch_data_samples (list[KIEDataSample]): A list of N datasamples,
                containing meta information and gold annotations for each of
                the images.

        Returns:
            List[KIEDataSample]: A list of datasamples of prediction results.
            Results are stored in ``pred_instances.labels`` and
            ``pred_instances.edge_labels``.
        """
        x = self.extract_feat(batch_inputs, [
            data_sample.gt_instances.bboxes
            for data_sample in batch_data_samples
        ])
        return self.kie_head.predict(x, batch_data_samples)

    def _forward(self, batch_inputs: torch.Tensor,
                 batch_data_samples: Sequence[KIEDataSample],
                 **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the raw tensor outputs from backbone and head without any post-
        processing.

        Args:
            batch_inputs (torch.Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            batch_data_samples (list[KIEDataSample]): A list of N datasamples,
                containing meta information and gold annotations for each of
                the images.

        Returns:
            tuple(torch.Tensor, torch.Tensor): Tensor output from head.

            - node_cls (torch.Tensor): Node classification output.
            - edge_cls (torch.Tensor): Edge classification output.
        """
        x = self.extract_feat(batch_inputs, [
            data_sample.gt_instances.bboxes
            for data_sample in batch_data_samples
        ])
        return self.kie_head(x, batch_data_samples)

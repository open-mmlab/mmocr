# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

from torch import Tensor

from mmocr.registry import MODELS, TASK_UTILS
from mmocr.utils import recog2spotting, spotting2recog
from mmocr.utils.typing import DetSampleList, OptMultiConfig
from .base import BaseRoIHead


@MODELS.register_module()
class OnlyRecRoIHead(BaseRoIHead):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self,
                 sampler: OptMultiConfig = None,
                 roi_extractor: OptMultiConfig = None,
                 rec_head: OptMultiConfig = None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if sampler is not None:
            self.sampler = TASK_UTILS.build(sampler)
        self.roi_extractor = MODELS.build(roi_extractor)
        self.rec_head = MODELS.build(rec_head)

    def loss(self, inputs: Tuple[Tensor], data_samples: DetSampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection
        roi on the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            DetSampleList (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """

        # assign gts and sample proposals
        data_samples = [self.sampler(ds) for ds in data_samples]

        bbox_feats = self.roi_extractor(inputs, data_samples)
        rec_data_samples = spotting2recog(data_samples, 'proposal')
        rec_loss = self.rec_head.loss(bbox_feats, rec_data_samples)

        return rec_loss

    def predict(self, inputs: Tuple[Tensor],
                data_samples: DetSampleList) -> DetSampleList:

        bbox_feats = self.roi_extractor(inputs, data_samples)
        rec_data_samples = spotting2recog(data_samples)
        rec_predicts = self.rec_head.predict(bbox_feats, rec_data_samples)
        return recog2spotting(rec_predicts, data_samples)

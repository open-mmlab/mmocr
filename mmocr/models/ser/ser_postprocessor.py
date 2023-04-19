# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import torch
import torch.nn as nn
from mmengine.structures import LabelData

from mmocr.registry import MODELS
from mmocr.structures import SERDataSample


@MODELS.register_module()
class SERPostprocessor(nn.Module):
    """PostProcessor for SER."""

    def __call__(self, outputs: torch.Tensor,
                 data_samples: Sequence[SERDataSample]
                 ) -> Sequence[SERDataSample]:
        outputs = outputs.cpu().detach()
        max_value, max_idx = torch.max(outputs, -1)
        for batch_idx in range(outputs.size(0)):
            pred_label = LabelData()
            pred_label.score = max_value[batch_idx]
            pred_label.item = max_idx[batch_idx]
            data_samples[batch_idx].pred_label = pred_label
        return data_samples

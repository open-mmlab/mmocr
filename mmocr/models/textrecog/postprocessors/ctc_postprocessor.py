# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Sequence, Tuple

import torch

from mmocr.data import TextRecogDataSample
from mmocr.registry import MODELS
from .base_textrecog_postprocessor import BaseTextRecogPostprocessor


# TODO support beam search
@MODELS.register_module()
class CTCPostProcessor(BaseTextRecogPostprocessor):
    """PostProcessor for CTC."""

    def get_single_prediction(self, probs: torch.Tensor,
                              data_sample: TextRecogDataSample
                              ) -> Tuple[Sequence[int], Sequence[float]]:
        """Convert the output probabilities of a single image to index and
        score.

        Args:
            probs (torch.Tensor): Character probabilities with shape
                :math:`(T, C)`.
            data_sample (TextRecogDataSample): Datasample of an image.

        Returns:
            tuple(list[int], list[float]): index and score.
        """
        feat_len = probs.size(0)
        max_value, max_idx = torch.max(probs, -1)
        valid_ratio = data_sample.get('valid_ratio', 1)
        decode_len = min(feat_len, math.ceil(feat_len * valid_ratio))
        index = []
        score = []

        prev_idx = self.dictionary.padding_idx
        for t in range(decode_len):
            tmp_value = max_idx[t].item()
            if tmp_value not in (prev_idx, *self.ignore_indexes):
                index.append(tmp_value)
                score.append(max_value[t].item())
            prev_idx = tmp_value
        return index, score

    def __call__(
        self, outputs: torch.Tensor,
        data_samples: Sequence[TextRecogDataSample]
    ) -> Sequence[TextRecogDataSample]:
        outputs = outputs.cpu().detach()
        return super().__call__(outputs, data_samples)

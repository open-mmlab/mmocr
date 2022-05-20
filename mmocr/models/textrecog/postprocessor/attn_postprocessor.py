# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple

import torch

from mmocr.core.data_structures import TextRecogDataSample
from mmocr.registry import MODELS
from .base_textrecog_postprocessor import BaseTextRecogPostprocessor


@MODELS.register_module()
class AttentionPostprocessor(BaseTextRecogPostprocessor):
    """PostProcessor for seq2seq."""

    def get_single_prediction(
        self,
        output: torch.Tensor,
        data_sample: Optional[TextRecogDataSample] = None,
    ) -> Tuple[Sequence[int], Sequence[float]]:
        """Convert the output of a single image to index and score.

        Args:
            output (torch.Tensor): Single image output.
            data_sample (TextRecogDataSample, optional): Datasample of an
                image. Defaults to None.

        Returns:
            tuple(list[int], list[float]): index and score.
        """
        max_value, max_idx = torch.max(output, -1)
        index, score = [], []
        output_index = max_idx.cpu().detach().numpy().tolist()
        output_score = max_value.cpu().detach().numpy().tolist()
        for char_index, char_score in zip(output_index, output_score):
            if char_index in self.ignore_indexes:
                continue
            if char_index == self.dictionary.end_idx:
                break
            index.append(char_index)
            score.append(char_score)
        return index, score

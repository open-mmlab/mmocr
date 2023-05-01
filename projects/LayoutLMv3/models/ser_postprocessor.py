# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Sequence, Union

import torch
import torch.nn as nn
from mmengine.structures import LabelData

from mmocr.registry import MODELS
from projects.LayoutLMv3.structures import SERDataSample
from projects.LayoutLMv3.utils.bio_label_utils import \
    find_other_label_name_of_biolabel


@MODELS.register_module()
class SERPostprocessor(nn.Module):
    """PostProcessor for SER."""

    def __init__(self,
                 classes: Union[tuple, list],
                 ignore_index: int = -100) -> None:
        super().__init__()
        self.other_label_name = find_other_label_name_of_biolabel(classes)
        self.id2biolabel = self._generate_id2biolabel_map(classes)
        self.ignore_index = ignore_index
        self.softmax = nn.Softmax(dim=-1)

    def _generate_id2biolabel_map(self, classes: Union[tuple, list]) -> Dict:
        bio_label_list = []
        classes = sorted([c for c in classes])
        for c in classes:
            if c == self.other_label_name:
                bio_label_list.insert(0, 'O')
            else:
                bio_label_list.append(f'B-{c}')
                bio_label_list.append(f'I-{c}')
        id2biolabel_map = {
            idx: bio_label
            for idx, bio_label in enumerate(bio_label_list)
        }
        return id2biolabel_map

    def __call__(self, outputs: torch.Tensor,
                 data_samples: Sequence[SERDataSample]
                 ) -> Sequence[SERDataSample]:
        # convert outputs dim from (truncation_num, max_length, label_num)
        # to (truncation_num * max_length, label_num)
        outputs = outputs.cpu().detach()
        truncation_num = outputs.size(0)
        outputs = torch.reshape(outputs, (-1, outputs.size(-1)))
        # merge gt label ids from data_samples
        gt_label_ids = [
            data_samples[truncation_idx].gt_label.item
            for truncation_idx in range(truncation_num)
        ]
        gt_label_ids = torch.cat(gt_label_ids, dim=0).cpu().detach().numpy()
        # get pred label ids/scores from outputs
        probs = self.softmax(outputs)
        max_value, max_idx = torch.max(probs, -1)
        pred_label_ids = max_idx.numpy()
        pred_label_scores = max_value.numpy()
        # select valid token and convert iid to biolabel
        gt_biolabels = [
            self.id2biolabel[g] for (g, p) in zip(gt_label_ids, pred_label_ids)
            if g != self.ignore_index
        ]
        pred_biolabels = [
            self.id2biolabel[p] for (g, p) in zip(gt_label_ids, pred_label_ids)
            if g != self.ignore_index
        ]
        pred_biolabel_scores = [
            s for (g, s) in zip(gt_label_ids, pred_label_scores)
            if g != self.ignore_index
        ]
        # record pred_label
        pred_label = LabelData()
        pred_label.item = pred_biolabels
        pred_label.score = pred_biolabel_scores
        # merge several truncation data_sample to one data_sample
        merged_data_sample = copy.deepcopy(data_samples[0])
        merged_data_sample.pred_label = pred_label
        # update merged gt_label
        merged_data_sample.gt_label.item = gt_biolabels
        return [merged_data_sample]

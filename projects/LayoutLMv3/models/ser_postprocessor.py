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

    def __init__(self, classes: Union[tuple, list]) -> None:
        super().__init__()
        self.other_label_name = find_other_label_name_of_biolabel(classes)
        self.id2biolabel = self._generate_id2biolabel_map(classes)
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
        # merge several truncation data_sample to one data_sample
        assert all('truncation_word_ids' in d for d in data_samples), \
            'The key `truncation_word_ids` should be specified' \
            'in PackSERInputs.'
        truncation_word_ids = []
        for data_sample in data_samples:
            truncation_word_ids.append(data_sample.pop('truncation_word_ids'))
        merged_data_sample = copy.deepcopy(data_samples[0])
        merged_data_sample.set_metainfo(
            dict(truncation_word_ids=truncation_word_ids))
        flattened_word_ids = [
            word_id for word_ids in truncation_word_ids for word_id in word_ids
        ]

        # convert outputs dim from (truncation_num, max_length, label_num)
        # to (truncation_num * max_length, label_num)
        outputs = outputs.cpu().detach()
        outputs = torch.reshape(outputs, (-1, outputs.size(-1)))
        # get pred label ids/scores from outputs
        probs = self.softmax(outputs)
        max_value, max_idx = torch.max(probs, -1)
        pred_label_ids = max_idx.numpy()
        pred_label_scores = max_value.numpy()

        # determine whether it is an inference process
        if 'item' in data_samples[0].gt_label:
            # merge gt label ids from data_samples
            gt_label_ids = [
                data_sample.gt_label.item for data_sample in data_samples
            ]
            gt_label_ids = torch.cat(
                gt_label_ids, dim=0).cpu().detach().numpy()
            gt_biolabels = [
                self.id2biolabel[g]
                for (w, g) in zip(flattened_word_ids, gt_label_ids)
                if w is not None
            ]
            # update merged gt_label
            merged_data_sample.gt_label.item = gt_biolabels

        # inference process do not have item in gt_label,
        # so select valid token with flattened_word_ids
        # rather than with gt_label_ids like official code.
        pred_biolabels = [
            self.id2biolabel[p]
            for (w, p) in zip(flattened_word_ids, pred_label_ids)
            if w is not None
        ]
        pred_biolabel_scores = [
            s for (w, s) in zip(flattened_word_ids, pred_label_scores)
            if w is not None
        ]
        # record pred_label
        pred_label = LabelData()
        pred_label.item = pred_biolabels
        pred_label.score = pred_biolabel_scores
        merged_data_sample.pred_label = pred_label

        return [merged_data_sample]

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
                 only_label_first_subword: bool = True) -> None:
        super().__init__()
        self.other_label_name = find_other_label_name_of_biolabel(classes)
        self.id2biolabel = self._generate_id2biolabel_map(classes)
        assert only_label_first_subword is True, \
            'Only support `only_label_first_subword=True` now.'
        self.only_label_first_subword = only_label_first_subword
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
        assert all('truncation_word_ids' in d for d in data_samples), \
            'The key `truncation_word_ids` should be specified' \
            'in PackSERInputs.'
        truncation_word_ids = [
            data_sample.pop('truncation_word_ids')
            for data_sample in data_samples
        ]
        word_ids = [
            word_id for word_ids in truncation_word_ids
            for word_id in word_ids[1:-1]
        ]

        # merge several truncation data_sample to one data_sample
        merged_data_sample = copy.deepcopy(data_samples[0])

        # convert outputs dim from (truncation_num, max_length, label_num)
        # to (truncation_num * max_length, label_num)
        outputs = outputs.cpu().detach()
        outputs = torch.reshape(outputs[:, 1:-1, :], (-1, outputs.size(-1)))
        # get pred label ids/scores from outputs
        probs = self.softmax(outputs)
        max_value, max_idx = torch.max(probs, -1)
        pred_label_ids = max_idx.numpy().tolist()
        pred_label_scores = max_value.numpy().tolist()

        # inference process do not have item in gt_label,
        # so select valid token with word_ids rather than
        # with gt_label_ids like official code.
        pred_words_biolabels = []
        word_biolabels = []
        pre_word_id = None
        for idx, cur_word_id in enumerate(word_ids):
            if cur_word_id is not None:
                if cur_word_id != pre_word_id:
                    if word_biolabels:
                        pred_words_biolabels.append(word_biolabels)
                        word_biolabels = []
                word_biolabels.append((self.id2biolabel[pred_label_ids[idx]],
                                       pred_label_scores[idx]))
            else:
                pred_words_biolabels.append(word_biolabels)
                break
            pre_word_id = cur_word_id
        if word_biolabels:
            pred_words_biolabels.append(word_biolabels)
        # record pred_label
        if self.only_label_first_subword:
            pred_label = LabelData()
            pred_label.item = [
                pred_word_biolabels[0][0]
                for pred_word_biolabels in pred_words_biolabels
            ]
            pred_label.score = [
                pred_word_biolabels[0][1]
                for pred_word_biolabels in pred_words_biolabels
            ]
            merged_data_sample.pred_label = pred_label
        else:
            raise NotImplementedError(
                'The `only_label_first_subword=False` is not support yet.')

        # determine whether it is an inference process
        if 'item' in data_samples[0].gt_label:
            # merge gt label ids from data_samples
            gt_label_ids = [
                data_sample.gt_label.item[1:-1] for data_sample in data_samples
            ]
            gt_label_ids = torch.cat(
                gt_label_ids, dim=0).cpu().detach().numpy().tolist()
            gt_words_biolabels = []
            word_biolabels = []
            pre_word_id = None
            for idx, cur_word_id in enumerate(word_ids):
                if cur_word_id is not None:
                    if cur_word_id != pre_word_id:
                        if word_biolabels:
                            gt_words_biolabels.append(word_biolabels)
                            word_biolabels = []
                    word_biolabels.append(self.id2biolabel[gt_label_ids[idx]])
                else:
                    gt_words_biolabels.append(word_biolabels)
                    break
                pre_word_id = cur_word_id
            if word_biolabels:
                gt_words_biolabels.append(word_biolabels)
            # update merged gt_label
            if self.only_label_first_subword:
                merged_data_sample.gt_label.item = [
                    gt_word_biolabels[0]
                    for gt_word_biolabels in gt_words_biolabels
                ]
            else:
                raise NotImplementedError(
                    'The `only_label_first_subword=False` is not support yet.')

        return [merged_data_sample]

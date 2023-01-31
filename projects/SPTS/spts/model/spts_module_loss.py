# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import Dict, Sequence, Union

import numpy as np
import torch
from torch import nn

from mmocr.models.common.dictionary import Dictionary
from mmocr.models.textrecog.module_losses import CEModuleLoss
from mmocr.registry import MODELS
from mmocr.structures import TextSpottingDataSample


@MODELS.register_module()
class SPTSModuleLoss(CEModuleLoss):
    """Implementation of loss module for SPTS with CrossEntropy loss.

    Args:
        dictionary (dict or :obj:`Dictionary`): The config for `Dictionary` or
            the instance of `Dictionary`.
        num_bins (int): Number of bins.
        seq_eos_coef (float): The loss weight coefficient of seq_eos token.
            Defaults to 0.01.
        max_seq_len (int): Maximum sequence length. The sequence is usually
            generated from decoder. Defaults to 40.
        letter_case (str): There are three options to alter the letter cases
            of gt texts:
            - unchanged: Do not change gt texts.
            - upper: Convert gt texts into uppercase characters.
            - lower: Convert gt texts into lowercase characters.
            Usually, it only works for English characters. Defaults to
            'unchanged'.
        pad_with (str): The padding strategy for ``gt_text.padded_indexes``.
            Defaults to 'auto'. Options are:
            - 'auto': Use dictionary.padding_idx to pad gt texts, or
              dictionary.end_idx if dictionary.padding_idx
              is None.
            - 'padding': Always use dictionary.padding_idx to pad gt texts.
            - 'end': Always use dictionary.end_idx to pad gt texts.
            - 'none': Do not pad gt texts.
        ignore_char (int or str): Specifies a target value that is
            ignored and does not contribute to the input gradient.
            ignore_char can be int or str. If int, it is the index of
            the ignored char. If str, it is the character to ignore.
            Apart from single characters, each item can be one of the
            following reversed keywords: 'padding', 'start', 'end',
            and 'unknown', which refer to their corresponding special
            tokens in the dictionary. It will not ignore any special
            tokens when ignore_char == -1 or 'none'. Defaults to 'padding'.
        flatten (bool): Whether to flatten the output and target before
            computing CE loss. Defaults to False.
        reduction (str): Specifies the reduction to apply to the output,
            should be one of the following: ('none', 'mean', 'sum'). Defaults
            to 'none'.
        ignore_first_char (bool): Whether to ignore the first token in target (
            usually the start token). If ``True``, the last token of the output
            sequence will also be removed to be aligned with the target length.
            Defaults to ``False``.
        flatten (bool): Whether to flatten the vectors for loss computation.
            Defaults to False.
    """

    def __init__(self,
                 dictionary: Union[Dict, Dictionary],
                 num_bins: int,
                 seq_eos_coef: float = 0.01,
                 max_seq_len: int = 40,
                 max_text_len: int = 25,
                 letter_case: str = 'unchanged',
                 pad_with: str = 'auto',
                 ignore_char: Union[int, str] = 'padding',
                 flatten: bool = False,
                 reduction: str = 'none',
                 ignore_first_char: bool = False):
        super().__init__(dictionary, max_seq_len, letter_case, pad_with,
                         ignore_char, flatten, reduction, ignore_first_char)
        # TODO: fix hardcode
        self.max_text_len = max_text_len
        self.max_num_text = (self.max_seq_len - 1) // (2 + max_text_len)
        self.num_bins = num_bins

        weights = torch.ones(self.dictionary.num_classes + num_bins)
        weights[self.dictionary.seq_end_idx] = seq_eos_coef
        weights.requires_grad_ = False
        self.loss_ce = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index,
            reduction=reduction,
            weight=weights)

    def get_targets(
        self, data_samples: Sequence[TextSpottingDataSample]
    ) -> Sequence[TextSpottingDataSample]:
        """Target generator.

        Args:
            data_samples (list[TextSpottingDataSample]): It usually includes
                ``gt_instances`` information.

        Returns:
            list[TextSpottingDataSample]: Updated data_samples. Two keys will
            be added to data_sample:

            - indexes (torch.LongTensor): Character indexes representing gt
              texts. All special tokens are excluded, except for UKN.
            - padded_indexes (torch.LongTensor): Character indexes
              representing gt texts with BOS and EOS if applicable, following
              several padding indexes until the length reaches ``max_seq_len``.
              In particular, if ``pad_with='none'``, no padding will be
              applied.
        """

        batch_max_len = 0

        for data_sample in data_samples:
            if data_sample.get('have_target', False):
                continue

            if len(data_sample.gt_instances.polygons) > self.max_num_text:
                keep = random.sample(
                    range(len(data_sample.gt_instances['polygons'])),
                    self.max_num_text)
                data_sample.gt_instances = data_sample.gt_instances[keep]

            gt_instances = data_sample.gt_instances

            if len(gt_instances.polygons) > 0:
                center_pts = []
                # Slightly different from the original implementation
                # which gets the center points from bezier curves
                for polygon in gt_instances.polygons:
                    center_pt = polygon.reshape(-1, 2).mean(0)
                    center_pts.append(center_pt)
                center_pts = np.vstack(center_pts)
                center_pts /= data_sample.img_shape[::-1]
                center_pts = torch.from_numpy(center_pts).type(torch.float32)
            else:
                center_pts = torch.ones(0).reshape(-1, 2).type(torch.float32)

            center_pts = (center_pts * self.num_bins).floor().type(torch.long)
            center_pts = torch.clamp(center_pts, min=0, max=self.num_bins - 1)

            gt_indexes = []
            for text in gt_instances.texts:
                if self.letter_case in ['upper', 'lower']:
                    text = getattr(text, self.letter_case)()

                indexes = self.dictionary.str2idx(text)
                indexes_tensor = torch.zeros(
                    self.max_text_len,
                    dtype=torch.long) + self.dictionary.end_idx
                max_len = min(self.max_text_len - 1, len(indexes))
                indexes_tensor[:max_len] = torch.LongTensor(indexes)[:max_len]
                indexes_tensor = indexes_tensor + self.num_bins
                gt_indexes.append(indexes_tensor)

            if len(gt_indexes) == 0:
                gt_indexes = torch.ones(0).reshape(-1, self.max_text_len)
            else:
                gt_indexes = torch.vstack(gt_indexes)
            gt_indexes = torch.cat([center_pts, gt_indexes], dim=-1)
            gt_indexes = gt_indexes.flatten()

            if self.dictionary.start_idx is not None:
                gt_indexes = torch.cat([
                    torch.LongTensor(
                        [self.dictionary.start_idx + self.num_bins]),
                    gt_indexes
                ])
            if self.dictionary.seq_end_idx is not None:
                gt_indexes = torch.cat([
                    gt_indexes,
                    torch.LongTensor(
                        [self.dictionary.seq_end_idx + self.num_bins])
                ])

            batch_max_len = max(batch_max_len, len(gt_indexes))

            gt_instances.set_metainfo(dict(indexes=gt_indexes))

        # Here we have to have the second pass as we need to know the max
        # length of the batch to pad the indexes in order to save memory
        for data_sample in data_samples:

            if data_sample.get('have_target', False):
                continue

            indexes = data_sample.gt_instances.indexes

            padded_indexes = (
                torch.zeros(batch_max_len, dtype=torch.long) +
                self.dictionary.padding_idx + self.num_bins)
            padded_indexes[:len(indexes)] = indexes
            data_sample.gt_instances.set_metainfo(
                dict(padded_indexes=padded_indexes))
            data_sample.set_metainfo(dict(have_target=True))

        return data_samples

    def forward(self, outputs: torch.Tensor,
                data_samples: Sequence[TextSpottingDataSample]) -> Dict:
        """
        Args:
            outputs (Tensor): A raw logit tensor of shape :math:`(N, T, C)`.
            data_samples (list[TextSpottingDataSample]): List of
                ``TextSpottingDataSample`` which are processed by
                ``get_targets``.

        Returns:
            dict: A loss dict with the key ``loss_ce``.
        """
        targets = list()
        for data_sample in data_samples:
            targets.append(data_sample.gt_instances.padded_indexes)
        targets = torch.stack(targets, dim=0).long()
        if self.ignore_first_char:
            targets = targets[:, 1:].contiguous()
            # outputs = outputs[:, :-1, :].contiguous()
        if self.flatten:
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
        else:
            outputs = outputs.permute(0, 2, 1).contiguous()

        loss_ce = self.loss_ce(outputs, targets.to(outputs.device))
        losses = dict(loss_ce=loss_ce)

        return losses

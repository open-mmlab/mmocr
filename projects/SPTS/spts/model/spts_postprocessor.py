# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from mmengine.structures import InstanceData

from mmocr.models import Dictionary
from mmocr.models.textrecog.postprocessors import BaseTextRecogPostprocessor
from mmocr.registry import MODELS
from mmocr.structures import TextSpottingDataSample
from mmocr.utils import rescale_polygons


@MODELS.register_module()
class SPTSPostprocessor(BaseTextRecogPostprocessor):
    """PostProcessor for SPTS.

    Args:
        rescale_fields (list[str], optional): The bbox/polygon field names to
            be rescaled. If None, no rescaling will be performed.
    """

    def __init__(self,
                 dictionary: Union[Dictionary, Dict],
                 num_bins: int,
                 rescale_fields: Optional[Sequence[str]] = ['points'],
                 max_seq_len: int = 40,
                 ignore_chars: Sequence[str] = ['padding'],
                 **kwargs) -> None:
        assert rescale_fields is None or isinstance(rescale_fields, list)
        self.num_bins = num_bins
        self.rescale_fields = rescale_fields
        super().__init__(
            dictionary=dictionary,
            num_bins=num_bins,
            max_seq_len=max_seq_len,
            ignore_chars=ignore_chars)

    def get_single_prediction(
        self,
        max_probs: torch.Tensor,
        seq: torch.Tensor,
        data_sample: Optional[TextSpottingDataSample] = None,
    ) -> Tuple[List[List[int]], List[List[float]], List[Tuple[float]],
               List[Tuple[float]]]:
        """Convert the output probabilities of a single image to character
        indexes, character scores, points and point scores.

        Args:
            max_probs (torch.Tensor): Character probabilities with shape
                :math:`(T)`.
            seq (torch.Tensor): Sequence indexes with shape
                :math:`(T)`.

            data_sample (TextSpottingDataSample, optional): Datasample of an
                image. Defaults to None.

        Returns:
            tuple(list[list[int]], list[list[float]], list[(float, float)],
            list(float, float)): character indexes, character scores, points
            and point scores. Each has len of max_seq_len.
        """
        h, w = data_sample.img_shape
        # the if is not a must since the softmaxed are masked out in decoder
        # if len(max_probs) % 27 != 0:
        #     max_probs = max_probs[:-len(max_probs) % 27]
        #     seq = seq[:-len(seq) % 27]
        # max_value, max_idx = torch.max(max_probs, -1)
        max_probs = max_probs.reshape(-1, 27)
        seq = seq.reshape(-1, 27)

        indexes, text_scores, points, pt_scores = [], [], [], []
        output_indexes = seq.cpu().detach().numpy().tolist()
        output_scores = max_probs.cpu().detach().numpy().tolist()
        for output_index, output_score in zip(output_indexes, output_scores):
            # work for multi-batch
            # if output_index[0] == self.dictionary.seq_end_idx +self.num_bins:
            #     break
            point_x = output_index[0] / self.num_bins * w
            point_y = output_index[1] / self.num_bins * h
            points.append((point_x, point_y))
            pt_scores.append(
                np.mean([
                    output_score[0],
                    output_score[1],
                ]).item())
            indexes.append([])
            char_scores = []
            for char_index, char_score in zip(output_index[2:],
                                              output_score[2:]):
                # the first num_bins indexes are for points
                dict_idx = char_index - self.num_bins
                if dict_idx in self.ignore_indexes:
                    continue
                if dict_idx == self.dictionary.end_idx:
                    break
                indexes[-1].append(dict_idx)
                char_scores.append(char_score)
            text_scores.append(np.mean(char_scores).item())
        return indexes, text_scores, points, pt_scores

    def __call__(
        self, output: Tuple[torch.Tensor, torch.Tensor],
        data_samples: Sequence[TextSpottingDataSample]
    ) -> Sequence[TextSpottingDataSample]:
        """Convert outputs to strings and scores.

        Args:
            TODO: fix docstr
            probs (torch.Tensor): Batched character probabilities, the model's
                softmaxed output in size: :math:`(N, T, C)`.
            data_samples (list[TextSpottingDataSample]): The list of
                TextSpottingDataSample.

        Returns:
            list(TextSpottingDataSample): The list of TextSpottingDataSample.
        """
        max_probs, seq = output
        batch_size = max_probs.size(0)

        for idx in range(batch_size):
            (char_idxs, text_scores, points,
             pt_scores) = self.get_single_prediction(max_probs[idx, :],
                                                     seq[idx, :],
                                                     data_samples[idx])
            texts = []
            scores = []
            for index, pt_score in zip(char_idxs, pt_scores):
                text = self.dictionary.idx2str(index)
                texts.append(text)
                # the "scores" field only accepts a float number
                scores.append(np.mean(pt_score).item())
            pred_instances = InstanceData()
            pred_instances.texts = texts
            pred_instances.scores = scores
            pred_instances.text_scores = text_scores
            pred_instances.points = points
            data_samples[idx].pred_instances = pred_instances
            pred_instances = self.rescale(data_samples[idx],
                                          data_samples[idx].scale_factor)
        return data_samples

    def rescale(self, results: TextSpottingDataSample,
                scale_factor: Sequence[int]) -> TextSpottingDataSample:
        """Rescale results in ``results.pred_instances`` according to
        ``scale_factor``, whose keys are defined in ``self.rescale_fields``.
        Usually used to rescale bboxes and/or polygons.

        Args:
            results (TextSpottingDataSample): The post-processed prediction
                results.
            scale_factor (tuple(int)): (w_scale, h_scale)

        Returns:
            TextDetDataSample: Prediction results with rescaled results.
        """
        scale_factor = np.asarray(scale_factor)
        for key in self.rescale_fields:
            # TODO: this util may need an alias or to be renamed
            results.pred_instances[key] = rescale_polygons(
                results.pred_instances[key], scale_factor, mode='div')
        return results

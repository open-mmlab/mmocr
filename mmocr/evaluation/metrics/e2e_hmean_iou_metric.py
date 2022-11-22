# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from shapely.geometry import Polygon

from mmocr.evaluation.functional import compute_hmean
from mmocr.registry import METRICS
from mmocr.utils import poly_intersection, poly_iou, polys2shapely


@METRICS.register_module()
class E2EHmeanIOUMetric(BaseMetric):
    """HmeanIOU metric.

    This method computes the hmean iou metric, which is done in the
    following steps:

    - Filter the prediction polygon:

      - Scores is smaller than minimum prediction score threshold.
      - The proportion of the area that intersects with gt ignored polygon is
        greater than ignore_precision_thr.

    - Computing an M x N IoU matrix, where each element indexing
      E_mn represents the IoU between the m-th valid GT and n-th valid
      prediction.
    - Based on different prediction score threshold:
      - Obtain the ignored predictions according to prediction score.
        The filtered predictions will not be involved in the later metric
        computations.
      - Based on the IoU matrix, get the match metric according to
      ``match_iou_thr``.
      - Based on different `strategy`, accumulate the match number.
    - calculate H-mean under different prediction score threshold.

    Args:
        match_iou_thr (float): IoU threshold for a match. Defaults to 0.5.
        ignore_precision_thr (float): Precision threshold when prediction and\
            gt ignored polygons are matched. Defaults to 0.5.
        pred_score_thrs (dict): Best prediction score threshold searching
            space. Defaults to dict(start=0.3, stop=0.9, step=0.1).
        strategy (str): Polygon matching strategy. Options are 'max_matching'
            and 'vanilla'. 'max_matching' refers to the optimum strategy that
            maximizes the number of matches. Vanilla strategy matches gt and
            pred polygons if both of them are never matched before. It was used
            in MMOCR 0.x and and academia. Defaults to 'vanilla'.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None
    """
    default_prefix: Optional[str] = 'icdar'

    def __init__(self,
                 match_iou_thr: float = 0.5,
                 ignore_precision_thr: float = 0.5,
                 pred_score_thrs: Dict = dict(start=0.3, stop=0.9, step=0.1),
                 lexicon_path: Optional[str] = None,
                 word_spotting: bool = False,
                 min_length_case_word: int = 3,
                 special_characters: str = "'!?.:,*\"()·[]/",
                 strategy: str = 'vanilla',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.match_iou_thr = match_iou_thr
        self.ignore_precision_thr = ignore_precision_thr
        self.pred_score_thrs = np.arange(**pred_score_thrs)
        self.word_spotting = word_spotting
        self.min_length_case_word = min_length_case_word
        self.special_characters = special_characters
        assert strategy in ['max_matching', 'vanilla']
        self.strategy = strategy

    def process(self, data_batch: Sequence[Dict],
                data_samples: Sequence[Dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[Dict]): A batch of data from dataloader.
            data_samples (Sequence[Dict]): A batch of outputs from
                the model.
        """
        for data_sample in data_samples:

            pred_instances = data_sample.get('pred_instances')
            pred_polygons = pred_instances.get('polygons')
            pred_scores = pred_instances.get('scores')
            if isinstance(pred_scores, torch.Tensor):
                pred_scores = pred_scores.cpu().numpy()
            pred_scores = np.array(pred_scores, dtype=np.float32)
            pred_texts = pred_instances.get('texts')

            gt_instances = data_sample.get('gt_instances')
            gt_polys = gt_instances.get('polygons')
            gt_ignore_flags = gt_instances.get('ignored')
            gt_texts = gt_instances.get('texts')
            if isinstance(gt_ignore_flags, torch.Tensor):
                gt_ignore_flags = gt_ignore_flags.cpu().numpy()
            gt_polys = polys2shapely(gt_polys)
            pred_polys = polys2shapely(pred_polygons)
            if self.word_spotting:
                gt_ignore_flags, gt_texts = self._word_spotting_filter(
                    gt_ignore_flags, gt_texts)
            pred_ignore_flags = self._filter_preds(pred_polys, gt_polys,
                                                   pred_scores,
                                                   gt_ignore_flags)
            pred_indexes = self._true_indexes(~pred_ignore_flags)
            gt_indexes = self._true_indexes(~gt_ignore_flags)
            pred_texts = [pred_texts[i] for i in pred_indexes]
            gt_texts = [gt_texts[i] for i in gt_indexes]

            gt_num = np.sum(~gt_ignore_flags)
            pred_num = np.sum(~pred_ignore_flags)
            iou_metric = np.zeros([gt_num, pred_num])

            # Compute IoU scores amongst kept pred and gt polygons
            for pred_mat_id, pred_poly_id in enumerate(pred_indexes):
                for gt_mat_id, gt_poly_id in enumerate(gt_indexes):
                    iou_metric[gt_mat_id, pred_mat_id] = poly_iou(
                        gt_polys[gt_poly_id], pred_polys[pred_poly_id])

            result = dict(
                gt_texts=gt_texts,
                pred_texts=pred_texts,
                iou_metric=iou_metric,
                pred_scores=pred_scores[~pred_ignore_flags])
            self.results.append(result)

    def compute_metrics(self, results: List[Dict]) -> Dict:
        """Compute the metrics from processed results.

        Args:
            results (list[dict]): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        best_eval_results = dict(hmean=-1)
        logger.info('Evaluating hmean-iou...')

        dataset_pred_num = np.zeros_like(self.pred_score_thrs)
        dataset_hit_num = np.zeros_like(self.pred_score_thrs)
        dataset_gt_num = 0
        for result in results:
            iou_metric = result['iou_metric']  # (gt_num, pred_num)
            pred_scores = result['pred_scores']  # (pred_num)
            gt_texts = result['gt_texts']
            pred_texts = result['pred_texts']
            dataset_gt_num += iou_metric.shape[0]

            # Filter out predictions by IoU threshold
            for i, pred_score_thr in enumerate(self.pred_score_thrs):
                pred_ignore_flags = pred_scores < pred_score_thr
                # get the number of matched boxes
                pred_texts = [
                    pred_texts[j]
                    for j in self._true_indexes(~pred_ignore_flags)
                ]
                matched_metric = iou_metric[:, ~pred_ignore_flags] \
                    > self.match_iou_thr
                if self.strategy == 'max_matching':
                    csr_matched_metric = csr_matrix(matched_metric)
                    matched_preds = maximum_bipartite_matching(
                        csr_matched_metric, perm_type='row')
                    # -1 denotes unmatched pred polygons
                    dataset_hit_num[i] += np.sum(matched_preds != -1)
                else:
                    # first come first matched
                    matched_gt_indexes = set()
                    matched_pred_indexes = set()
                    matched_e2e_gt_indexes = set()
                    for gt_idx, pred_idx in zip(*np.nonzero(matched_metric)):
                        if gt_idx in matched_gt_indexes or \
                          pred_idx in matched_pred_indexes:
                            continue
                        matched_gt_indexes.add(gt_idx)
                        matched_pred_indexes.add(pred_idx)
                        if self.word_spotting:
                            if gt_texts[gt_idx] == pred_texts[pred_idx]:
                                matched_e2e_gt_indexes.add(gt_idx)
                        else:
                            if self.text_match(gt_texts[gt_idx].upper(),
                                               pred_texts[pred_idx].upper()):
                                matched_e2e_gt_indexes.add(gt_idx)
                    dataset_hit_num[i] += len(matched_e2e_gt_indexes)
                dataset_pred_num[i] += np.sum(~pred_ignore_flags)

        for i, pred_score_thr in enumerate(self.pred_score_thrs):
            recall, precision, hmean = compute_hmean(
                int(dataset_hit_num[i]), int(dataset_hit_num[i]),
                int(dataset_gt_num), int(dataset_pred_num[i]))
            eval_results = dict(
                precision=precision, recall=recall, hmean=hmean)
            logger.info(f'prediction score threshold: {pred_score_thr:.2f}, '
                        f'recall: {eval_results["recall"]:.4f}, '
                        f'precision: {eval_results["precision"]:.4f}, '
                        f'hmean: {eval_results["hmean"]:.4f}\n')
            if eval_results['hmean'] > best_eval_results['hmean']:
                best_eval_results = eval_results
        return best_eval_results

    def _filter_preds(self, pred_polys: List[Polygon], gt_polys: List[Polygon],
                      pred_scores: List[float],
                      gt_ignore_flags: np.ndarray) -> np.ndarray:
        """Filter out the predictions by score threshold and whether it
        overlaps ignored gt polygons.

        Args:
            pred_polys (list[Polygon]): Pred polygons.
            gt_polys (list[Polygon]): GT polygons.
            pred_scores (list[float]): Pred scores of polygons.
            gt_ignore_flags (np.ndarray): 1D boolean array indicating
                the positions of ignored gt polygons.

        Returns:
            np.ndarray: 1D boolean array indicating the positions of ignored
            pred polygons.
        """

        # Filter out predictions based on the minimum score threshold
        pred_ignore_flags = pred_scores < self.pred_score_thrs.min()
        pred_indexes = self._true_indexes(~pred_ignore_flags)
        gt_indexes = self._true_indexes(gt_ignore_flags)
        # Filter out pred polygons which overlaps any ignored gt polygons
        for pred_id in pred_indexes:
            for gt_id in gt_indexes:
                # Match pred with ignored gt
                precision = poly_intersection(
                    gt_polys[gt_id], pred_polys[pred_id]) / (
                        pred_polys[pred_id].area + 1e-5)
                if precision > self.ignore_precision_thr:
                    pred_ignore_flags[pred_id] = True
                    break

        return pred_ignore_flags

    def _true_indexes(self, array: np.ndarray) -> np.ndarray:
        """Get indexes of True elements from a 1D boolean array."""
        return np.where(array)[0]

    def _include_in_dictionary(self, text):
        """Function used in Word Spotting that finds if the Ground Truth text
        meets the rules to enter into the dictionary.

        If not, the text will be cared as don't care
        """
        # special case 's at final
        if text[len(text) - 2:] == "'s" or text[len(text) - 2:] == "'S":
            text = text[0:len(text) - 2]

        # hyphens at init or final of the word
        text = text.strip('-')

        for character in self.special_characters:
            text = text.replace(character, ' ')

        text = text.strip()

        if len(text) != len(text.replace(' ', '')):
            return False

        if len(text) < self.min_length_case_word:
            return False

        notAllowed = '×÷·'

        range1 = [ord(u'a'), ord(u'z')]
        range2 = [ord(u'A'), ord(u'Z')]
        range3 = [ord(u'À'), ord(u'ƿ')]
        range4 = [ord(u'Ǆ'), ord(u'ɿ')]
        range5 = [ord(u'Ά'), ord(u'Ͽ')]
        range6 = [ord(u'-'), ord(u'-')]

        for char in text:
            charCode = ord(char)
            if (notAllowed.find(char) != -1):
                return False

            valid = (charCode >= range1[0] and charCode <= range1[1]) or (
                charCode >= range2[0] and charCode <= range2[1]
            ) or (charCode >= range3[0] and charCode <= range3[1]) or (
                charCode >= range4[0] and charCode <= range4[1]) or (
                    charCode >= range5[0]
                    and charCode <= range5[1]) or (charCode >= range6[0]
                                                   and charCode <= range6[1])
            if not valid:
                return False

        return True

    def _include_in_dictionary_text(self, text):
        """Function applied to the Ground Truth texts used in Word Spotting.

        It removes special characters or terminations
        """
        # special case 's at final
        if text[len(text) - 2:] == "'s" or text[len(text) - 2:] == "'S":
            text = text[0:len(text) - 2]

        # hyphens at init or final of the word
        text = text.strip('-')

        for character in self.special_characters:
            text = text.replace(character, ' ')

        text = text.strip()

        return text

    def text_match(self,
                   gt_text,
                   pred_text,
                   only_remove_first_end_character=True):

        if only_remove_first_end_character:
            # special characters in GT are allowed only at initial or final
            # position
            if (gt_text == pred_text):
                return True

            if self.special_characters.find(gt_text[0]) > -1:
                if gt_text[1:] == pred_text:
                    return True

            if self.special_characters.find(gt_text[-1]) > -1:
                if gt_text[0:len(gt_text) - 1] == pred_text:
                    return True

            if self.special_characters.find(
                    gt_text[0]) > -1 and self.special_characters.find(
                        gt_text[-1]) > -1:
                if gt_text[1:len(gt_text) - 1] == pred_text:
                    return True
            return False
        else:
            # Special characters are removed from the beginning and the end of
            # both Detection and GroundTruth
            while len(gt_text) > 0 and self.special_characters.find(
                    gt_text[0]) > -1:
                gt_text = gt_text[1:]

            while len(pred_text) > 0 and self.special_characters.find(
                    pred_text[0]) > -1:
                pred_text = pred_text[1:]

            while len(gt_text) > 0 and self.special_characters.find(
                    gt_text[-1]) > -1:
                gt_text = gt_text[0:len(gt_text) - 1]

            while len(pred_text) > 0 and self.special_characters.find(
                    pred_text[-1]) > -1:
                pred_text = pred_text[0:len(pred_text) - 1]

            return gt_text == pred_text

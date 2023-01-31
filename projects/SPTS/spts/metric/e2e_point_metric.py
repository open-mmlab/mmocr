# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from shapely.geometry import LineString, Point

from mmocr.registry import METRICS


@METRICS.register_module()
class E2EPointMetric(BaseMetric):
    # TODO docstring
    """Point metric for textspotting. Proposed in SPTS.

    Args:
        ignore_precision_thr (float): Precision threshold when prediction and\
            gt ignored polygons are matched. Defaults to 0.5.
        text_score_thrs (dict): Best text score threshold searching
            space. Defaults to dict(start=0.8, stop=1, step=0.01).
        TODO: docstr
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None
    """
    default_prefix: Optional[str] = 'e2e_icdar'

    def __init__(self,
                 match_iou_thr: float = 0.5,
                 ignore_precision_thr: float = 0.5,
                 text_score_thrs: Dict = dict(start=0.8, stop=1, step=0.01),
                 lexicon_path: Optional[str] = None,
                 word_spotting: bool = False,
                 min_length_case_word: int = 3,
                 special_characters: str = "'!?.:,*\"()·[]/",
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.match_iou_thr = match_iou_thr
        self.ignore_precision_thr = ignore_precision_thr
        self.text_score_thrs = np.arange(**text_score_thrs)
        self.word_spotting = word_spotting
        self.min_length_case_word = min_length_case_word
        self.special_characters = special_characters

    def poly_center(self, poly_pts):
        poly_pts = np.array(poly_pts).reshape(-1, 2)
        num_points = poly_pts.shape[0]
        line1 = LineString(poly_pts[int(num_points / 2):])
        line2 = LineString(poly_pts[:int(num_points / 2)])
        mid_pt1 = np.array(line1.interpolate(0.5, normalized=True).coords[0])
        mid_pt2 = np.array(line2.interpolate(0.5, normalized=True).coords[0])
        return (mid_pt1 + mid_pt2) / 2

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
            pred_points = pred_instances.get('points')
            text_scores = pred_instances.get('text_scores')
            if isinstance(text_scores, torch.Tensor):
                text_scores = text_scores.cpu().numpy()
            text_scores = np.array(text_scores, dtype=np.float32)
            pred_texts = pred_instances.get('texts')

            gt_instances = data_sample.get('gt_instances')
            gt_polys = gt_instances.get('polygons')
            gt_ignore_flags = gt_instances.get('ignored')
            gt_texts = gt_instances.get('texts')
            if isinstance(gt_ignore_flags, torch.Tensor):
                gt_ignore_flags = gt_ignore_flags.cpu().numpy()

            gt_points = [self.poly_center(poly) for poly in gt_polys]
            if self.word_spotting:
                gt_ignore_flags, gt_texts = self._word_spotting_filter(
                    gt_ignore_flags, gt_texts)

            pred_ignore_flags = text_scores < self.text_score_thrs.min()
            text_scores = text_scores[~pred_ignore_flags]
            pred_texts = self._get_true_elements(pred_texts,
                                                 ~pred_ignore_flags)
            pred_points = self._get_true_elements(pred_points,
                                                  ~pred_ignore_flags)

            result = dict(
                text_scores=text_scores,
                pred_points=pred_points,
                gt_points=gt_points,
                pred_texts=pred_texts,
                gt_texts=gt_texts,
                gt_ignore_flags=gt_ignore_flags)
            self.results.append(result)

    def _get_true_elements(self, array: List, flags: np.ndarray) -> List:
        return [array[i] for i in self._true_indexes(flags)]

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

        num_thres = len(self.text_score_thrs)
        num_preds = np.zeros(
            num_thres, dtype=int)  # the number of points actually predicted
        num_tp = np.zeros(num_thres, dtype=int)  # number of true positives
        num_gts = np.zeros(num_thres, dtype=int)  # number of valid gts

        for result in results:
            text_scores = result['text_scores']
            pred_points = result['pred_points']
            gt_points = result['gt_points']
            gt_texts = result['gt_texts']
            pred_texts = result['pred_texts']
            gt_ignore_flags = result['gt_ignore_flags']

            # Filter out predictions by IoU threshold
            for i, text_score_thr in enumerate(self.text_score_thrs):
                pred_ignore_flags = text_scores < text_score_thr
                filtered_pred_texts = self._get_true_elements(
                    pred_texts, ~pred_ignore_flags)
                filtered_pred_points = self._get_true_elements(
                    pred_points, ~pred_ignore_flags)
                gt_matched = np.zeros(len(gt_texts), dtype=bool)
                num_gt = len(gt_texts) - np.sum(gt_ignore_flags)
                if num_gt == 0:
                    continue
                num_gts[i] += num_gt

                for pred_text, pred_point in zip(filtered_pred_texts,
                                                 filtered_pred_points):
                    dists = [
                        Point(pred_point).distance(Point(gt_point))
                        for gt_point in gt_points
                    ]
                    min_idx = np.argmin(dists)
                    if gt_texts[min_idx] == '###' or gt_ignore_flags[min_idx]:
                        continue
                    # if not gt_matched[min_idx] and self.text_match(
                    #         gt_texts[min_idx].upper(), pred_text.upper()):
                    if (not gt_matched[min_idx] and gt_texts[min_idx].upper()
                            == pred_text.upper()):
                        gt_matched[min_idx] = True
                        num_tp[i] += 1
                    num_preds[i] += 1

        for i, text_score_thr in enumerate(self.text_score_thrs):
            if num_preds[i] == 0 or num_tp[i] == 0:
                recall, precision, hmean = 0, 0, 0
            else:
                recall = num_tp[i] / num_gts[i]
                precision = num_tp[i] / num_preds[i]
                hmean = 2 * recall * precision / (recall + precision)
            eval_results = dict(
                precision=precision, recall=recall, hmean=hmean)
            logger.info(f'text score threshold: {text_score_thr:.2f}, '
                        f'recall: {eval_results["recall"]:.4f}, '
                        f'precision: {eval_results["precision"]:.4f}, '
                        f'hmean: {eval_results["hmean"]:.4f}\n')
            if eval_results['hmean'] > best_eval_results['hmean']:
                best_eval_results = eval_results
        return best_eval_results

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
            # TODO: optimize it with for loop
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

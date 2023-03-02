# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from rapidfuzz.distance import Levenshtein
from shapely.geometry import Point

from mmocr.registry import METRICS

# TODO: CTW1500 read pair


@METRICS.register_module()
class E2EPointMetric(BaseMetric):
    """Point metric for textspotting. Proposed in SPTS.

    Args:
        text_score_thrs (dict): Best text score threshold searching
            space. Defaults to dict(start=0.8, stop=1, step=0.01).
        word_spotting (bool): Whether to work in word spotting mode. Defaults
            to False.
        lexicon_path (str, optional): Lexicon path for word spotting, which
            points to a lexicon file or a directory. Defaults to None.
        lexicon_mapping (tuple, optional): The rule to map test image name to
            its corresponding lexicon file. Only effective when lexicon path
            is a directory. Defaults to ('(.*).jpg', r'\1.txt').
        pair_path (str, optional): Pair path for word spotting, which points
            to a pair file or a directory. Defaults to None.
        pair_mapping (tuple, optional): The rule to map test image name to
            its corresponding pair file. Only effective when pair path is a
            directory. Defaults to ('(.*).jpg', r'\1.txt').
        match_dist_thr (float, optional): Matching distance threshold for
            word spotting. Defaults to None.
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
                 text_score_thrs: Dict = dict(start=0.8, stop=1, step=0.01),
                 word_spotting: bool = False,
                 lexicon_path: Optional[str] = None,
                 lexicon_mapping: Tuple[str, str] = ('(.*).jpg', r'\1.txt'),
                 pair_path: Optional[str] = None,
                 pair_mapping: Tuple[str, str] = ('(.*).jpg', r'\1.txt'),
                 match_dist_thr: Optional[float] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.text_score_thrs = np.arange(**text_score_thrs)
        self.word_spotting = word_spotting
        self.match_dist_thr = match_dist_thr
        if lexicon_path:
            self.lexicon_mapping = lexicon_mapping
            self.pair_mapping = pair_mapping
            self.lexicons = self._read_lexicon(lexicon_path)
            self.pairs = self._read_pair(pair_path)

    def _read_lexicon(self, lexicon_path: str) -> List[str]:
        if lexicon_path.endswith('.txt'):
            lexicon = open(lexicon_path, 'r').read().splitlines()
            lexicon = [ele.strip() for ele in lexicon]
        else:
            lexicon = {}
            for file in glob.glob(osp.join(lexicon_path, '*.txt')):
                basename = osp.basename(file)
                lexicon[basename] = self._read_lexicon(file)
        return lexicon

    def _read_pair(self, pair_path: str) -> Dict[str, str]:
        pairs = {}
        if pair_path.endswith('.txt'):
            pair_lines = open(pair_path, 'r').read().splitlines()
            for line in pair_lines:
                line = line.strip()
                word = line.split(' ')[0].upper()
                word_gt = line[len(word) + 1:]
                pairs[word] = word_gt
        else:
            for file in glob.glob(osp.join(pair_path, '*.txt')):
                basename = osp.basename(file)
                pairs[basename] = self._read_pair(file)
        return pairs

    def poly_center(self, poly_pts):
        poly_pts = np.array(poly_pts).reshape(-1, 2)
        return poly_pts.mean(0)

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
                # reserved for image-level lexcions
                gt_img_name=osp.basename(data_sample.get('img_path', '')),
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
            gt_img_name = result['gt_img_name']

            # Correct the words with lexicon
            pred_dist_flags = np.zeros(len(pred_texts), dtype=bool)
            if hasattr(self, 'lexicons'):
                for i, pred_text in enumerate(pred_texts):
                    # If it's an image-level lexicon
                    if isinstance(self.lexicons, dict):
                        lexicon_name = self._map_img_name(
                            gt_img_name, self.lexicon_mapping)
                        pair_name = self._map_img_name(gt_img_name,
                                                       self.pair_mapping)
                        pred_texts[i], match_dist = self._match_word(
                            pred_text, self.lexicons[lexicon_name],
                            self.pairs[pair_name])
                    else:
                        pred_texts[i], match_dist = self._match_word(
                            pred_text, self.lexicons, self.pairs)
                    if (self.match_dist_thr
                            and match_dist >= self.match_dist_thr):
                        # won't even count this as a prediction
                        pred_dist_flags[i] = True

            # Filter out predictions by IoU threshold
            for i, text_score_thr in enumerate(self.text_score_thrs):
                pred_ignore_flags = pred_dist_flags | (
                    text_scores < text_score_thr)
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
                    if not gt_matched[min_idx] and (
                            pred_text.upper() == gt_texts[min_idx].upper()):
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

    def _map_img_name(self, img_name: str, mapping: Tuple[str, str]) -> str:
        """Map the image name to the another one based on mapping."""
        return re.sub(mapping[0], mapping[1], img_name)

    def _true_indexes(self, array: np.ndarray) -> np.ndarray:
        """Get indexes of True elements from a 1D boolean array."""
        return np.where(array)[0]

    def _word_spotting_filter(self, gt_ignore_flags: np.ndarray,
                              gt_texts: List[str]
                              ) -> Tuple[np.ndarray, List[str]]:
        """Filter out gt instances that cannot be in a valid dictionary, and do
        some simple preprocessing to texts."""

        for i in range(len(gt_texts)):
            if gt_ignore_flags[i]:
                continue
            text = gt_texts[i]
            if text[-2:] in ["'s", "'S"]:
                text = text[:-2]
            text = text.strip('-')
            for char in "'!?.:,*\"()·[]/":
                text = text.replace(char, ' ')
            text = text.strip()
            gt_ignore_flags[i] = not self._include_in_dict(text)
            if not gt_ignore_flags[i]:
                gt_texts[i] = text

        return gt_ignore_flags, gt_texts

    def _include_in_dict(self, text: str) -> bool:
        """Check if the text could be in a valid dictionary."""
        if len(text) != len(text.replace(' ', '')) or len(text) < 3:
            return False
        not_allowed = '×÷·'
        valid_ranges = [(ord(u'a'), ord(u'z')), (ord(u'A'), ord(u'Z')),
                        (ord(u'À'), ord(u'ƿ')), (ord(u'Ǆ'), ord(u'ɿ')),
                        (ord(u'Ά'), ord(u'Ͽ')), (ord(u'-'), ord(u'-'))]
        for char in text:
            code = ord(char)
            if (not_allowed.find(char) != -1):
                return False
            valid = any(code >= r[0] and code <= r[1] for r in valid_ranges)
            if not valid:
                return False
        return True

    def _match_word(self,
                    text: str,
                    lexicons: List[str],
                    pairs: Optional[Dict[str, str]] = None) -> Tuple[str, int]:
        """Match the text with the lexicons and pairs."""
        text = text.upper()
        matched_word = ''
        matched_dist = 100
        for word in lexicons:
            word = word.upper()
            norm_dist = Levenshtein.normalized_distance(text, word)
            if norm_dist < matched_dist:
                matched_dist = norm_dist
                if pairs:
                    matched_word = pairs[word]
                else:
                    matched_word = word
        return matched_word, matched_dist

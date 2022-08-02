# Copyright (c) OpenMMLab. All rights reserved.
import re
from difflib import SequenceMatcher
from typing import Dict, Optional, Sequence, Union

import mmcv
from mmengine.evaluator import BaseMetric
from rapidfuzz import string_metric

from mmocr.registry import METRICS


@METRICS.register_module()
class WordMetric(BaseMetric):
    """Word metrics for text recognition task.

    Args:
        mode (str or list[str]): Options are:
            - 'exact': Accuracy at word level.
            - 'ignore_case': Accuracy at word level, ignoring letter
              case.
            - 'ignore_case_symbol': Accuracy at word level, ignoring
              letter case and symbol. (Default metric for academic evaluation)
            If mode is a list, then metrics in mode will be calculated
            separately. Defaults to 'ignore_case_symbol'
        valid_symbol (str): Valid characters. Defaults to
            '[^A-Z^a-z^0-9^\u4e00-\u9fa5]'
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    default_prefix: Optional[str] = 'recog'

    def __init__(self,
                 mode: Union[str, Sequence[str]] = 'ignore_case_symbol',
                 valid_symbol: str = '[^A-Z^a-z^0-9^\u4e00-\u9fa5]',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device, prefix)
        self.valid_symbol = re.compile(valid_symbol)
        if isinstance(mode, str):
            mode = [mode]
        assert mmcv.is_seq_of(mode, str)
        assert set(mode).issubset(
            {'exact', 'ignore_case', 'ignore_case_symbol'})
        self.mode = set(mode)

    def process(self, data_batch: Sequence[Dict],
                predictions: Sequence[Dict]) -> None:
        """Process one batch of predictions. The processed results should be
        stored in ``self.results``, which will be used to compute the metrics
        when all batches have been processed.

        Args:
            data_batch (Sequence[Dict]): A batch of gts.
            predictions (Sequence[Dict]): A batch of outputs from the model.
        """
        for data_sample in predictions:
            match_num = 0
            match_ignore_case_num = 0
            match_ignore_case_symbol_num = 0
            pred_text = data_sample.get('pred_text').get('item')
            gt_text = data_sample.get('gt_text').get('item')
            if 'ignore_case' in self.mode or 'ignore_case_symbol' in self.mode:
                pred_text_lower = pred_text.lower()
                gt_text_lower = gt_text.lower()
            if 'ignore_case_symbol' in self.mode:
                gt_text_lower_ignore = self.valid_symbol.sub('', gt_text_lower)
                pred_text_lower_ignore = self.valid_symbol.sub(
                    '', pred_text_lower)
                match_ignore_case_symbol_num =\
                    gt_text_lower_ignore == pred_text_lower_ignore
            if 'ignore_case' in self.mode:
                match_ignore_case_num = pred_text_lower == gt_text_lower
            if 'exact' in self.mode:
                match_num = pred_text == gt_text
            result = dict(
                match_num=match_num,
                match_ignore_case_num=match_ignore_case_num,
                match_ignore_case_symbol_num=match_ignore_case_symbol_num)
            self.results.append(result)

    def compute_metrics(self, results: Sequence[Dict]) -> Dict:
        """Compute the metrics from processed results.

        Args:
            results (list[Dict]): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

        eps = 1e-8
        eval_res = {}
        gt_word_num = len(results)
        if 'exact' in self.mode:
            match_nums = [result['match_num'] for result in results]
            match_nums = sum(match_nums)
            eval_res['word_acc'] = 1.0 * match_nums / (eps + gt_word_num)
        if 'ignore_case' in self.mode:
            match_ignore_case_num = [
                result['match_ignore_case_num'] for result in results
            ]
            match_ignore_case_num = sum(match_ignore_case_num)
            eval_res['word_acc_ignore_case'] = 1.0 *\
                match_ignore_case_num / (eps + gt_word_num)
        if 'ignore_case_symbol' in self.mode:
            match_ignore_case_symbol_num = [
                result['match_ignore_case_symbol_num'] for result in results
            ]
            match_ignore_case_symbol_num = sum(match_ignore_case_symbol_num)
            eval_res['word_acc_ignore_case_symbol'] = 1.0 *\
                match_ignore_case_symbol_num / (eps + gt_word_num)

        for key, value in eval_res.items():
            eval_res[key] = float(f'{value:.4f}')
        return eval_res


@METRICS.register_module()
class CharMetric(BaseMetric):
    """Character metrics for text recognition task.

    Args:
        valid_symbol (str): Valid characters.
            Defaults to '[^A-Z^a-z^0-9^\u4e00-\u9fa5]'
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    default_prefix: Optional[str] = 'recog'

    def __init__(self,
                 valid_symbol: str = '[^A-Z^a-z^0-9^\u4e00-\u9fa5]',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device, prefix)
        self.valid_symbol = re.compile(valid_symbol)

    def process(self, data_batch: Sequence[Dict],
                predictions: Sequence[Dict]) -> None:
        """Process one batch of predictions. The processed results should be
        stored in ``self.results``, which will be used to compute the metrics
        when all batches have been processed.

        Args:
            data_batch (Sequence[Dict]): A batch of gts.
            predictions (Sequence[Dict]): A batch of outputs from the model.
        """
        for data_sample in predictions:
            pred_text = data_sample.get('pred_text').get('item')
            gt_text = data_sample.get('gt_text').get('item')
            gt_text_lower = gt_text.lower()
            pred_text_lower = pred_text.lower()
            gt_text_lower_ignore = self.valid_symbol.sub('', gt_text_lower)
            pred_text_lower_ignore = self.valid_symbol.sub('', pred_text_lower)
            # number to calculate char level recall & precision
            result = dict(
                gt_char_num=len(gt_text_lower_ignore),
                pred_char_num=len(pred_text_lower_ignore),
                true_positive_char_num=self._cal_true_positive_char(
                    pred_text_lower_ignore, gt_text_lower_ignore))
            self.results.append(result)

    def compute_metrics(self, results: Sequence[Dict]) -> Dict:
        """Compute the metrics from processed results.

        Args:
            results (list[Dict]): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the
            metrics, and the values are corresponding results.
        """
        gt_char_num = [result['gt_char_num'] for result in results]
        pred_char_num = [result['pred_char_num'] for result in results]
        true_positive_char_num = [
            result['true_positive_char_num'] for result in results
        ]
        gt_char_num = sum(gt_char_num)
        pred_char_num = sum(pred_char_num)
        true_positive_char_num = sum(true_positive_char_num)

        eps = 1e-8
        char_recall = 1.0 * true_positive_char_num / (eps + gt_char_num)
        char_precision = 1.0 * true_positive_char_num / (eps + pred_char_num)
        eval_res = {}
        eval_res['char_recall'] = char_recall
        eval_res['char_precision'] = char_precision

        for key, value in eval_res.items():
            eval_res[key] = float(f'{value:.4f}')
        return eval_res

    def _cal_true_positive_char(self, pred: str, gt: str) -> int:
        """Calculate correct character number in prediction.

        Args:
            pred (str): Prediction text.
            gt (str): Ground truth text.

        Returns:
            true_positive_char_num (int): The true positive number.
        """

        all_opt = SequenceMatcher(None, pred, gt)
        true_positive_char_num = 0
        for opt, _, _, s2, e2 in all_opt.get_opcodes():
            if opt == 'equal':
                true_positive_char_num += (e2 - s2)
            else:
                pass
        return true_positive_char_num


@METRICS.register_module()
class OneMinusNEDMetric(BaseMetric):
    """One minus NED metric for text recognition task.

    Args:
        valid_symbol (str): Valid characters. Defaults to
            '[^A-Z^a-z^0-9^\u4e00-\u9fa5]'
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None
    """
    default_prefix: Optional[str] = 'recog'

    def __init__(self,
                 valid_symbol: str = '[^A-Z^a-z^0-9^\u4e00-\u9fa5]',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device, prefix)
        self.valid_symbol = re.compile(valid_symbol)

    def process(self, data_batch: Sequence[Dict],
                predictions: Sequence[Dict]) -> None:
        """Process one batch of predictions. The processed results should be
        stored in ``self.results``, which will be used to compute the metrics
        when all batches have been processed.

        Args:
            data_batch (Sequence[Dict]): A batch of gts.
            predictions (Sequence[Dict]): A batch of outputs from the model.
        """
        for data_sample in predictions:
            pred_text = data_sample.get('pred_text').get('item')
            gt_text = data_sample.get('gt_text').get('item')
            gt_text_lower = gt_text.lower()
            pred_text_lower = pred_text.lower()
            gt_text_lower_ignore = self.valid_symbol.sub('', gt_text_lower)
            pred_text_lower_ignore = self.valid_symbol.sub('', pred_text_lower)
            edit_dist = string_metric.levenshtein(pred_text_lower_ignore,
                                                  gt_text_lower_ignore)
            norm_ed = float(edit_dist) / max(1, len(gt_text_lower_ignore),
                                             len(pred_text_lower_ignore))
            result = dict(norm_ed=norm_ed)
            self.results.append(result)

    def compute_metrics(self, results: Sequence[Dict]) -> Dict:
        """Compute the metrics from processed results.

        Args:
            results (list[Dict]): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the
            metrics, and the values are corresponding results.
        """

        gt_word_num = len(results)
        norm_ed = [result['norm_ed'] for result in results]
        norm_ed_sum = sum(norm_ed)
        normalized_edit_distance = norm_ed_sum / max(1, gt_word_num)
        eval_res = {}
        eval_res['1-N.E.D'] = 1.0 - normalized_edit_distance
        for key, value in eval_res.items():
            eval_res[key] = float(f'{value:.4f}')
        return eval_res

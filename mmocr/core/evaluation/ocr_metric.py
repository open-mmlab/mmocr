# Copyright (c) OpenMMLab. All rights reserved.
import re
from difflib import SequenceMatcher

from rapidfuzz import string_metric

from mmocr.utils import is_type_list


def cal_true_positive_char(pred, gt):
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


def count_matches(pred_texts, gt_texts):
    """Count the various match number for metric calculation.

    Args:
        pred_texts (list[str]): Predicted text string.
        gt_texts (list[str]): Ground truth text string.

    Returns:
        match_res: (dict[str: int]): Match number used for
            metric calculation.
    """
    match_res = {
        'gt_char_num': 0,
        'pred_char_num': 0,
        'true_positive_char_num': 0,
        'gt_word_num': 0,
        'match_word_num': 0,
        'match_word_ignore_case': 0,
        'match_word_ignore_case_symbol': 0
    }
    comp = re.compile('[^A-Z^a-z^0-9^\u4e00-\u9fa5]')
    norm_ed_sum = 0.0
    for pred_text, gt_text in zip(pred_texts, gt_texts):
        if gt_text == pred_text:
            match_res['match_word_num'] += 1
        gt_text_lower = gt_text.lower()
        pred_text_lower = pred_text.lower()
        if gt_text_lower == pred_text_lower:
            match_res['match_word_ignore_case'] += 1
        gt_text_lower_ignore = comp.sub('', gt_text_lower)
        pred_text_lower_ignore = comp.sub('', pred_text_lower)
        if gt_text_lower_ignore == pred_text_lower_ignore:
            match_res['match_word_ignore_case_symbol'] += 1
        match_res['gt_word_num'] += 1

        # normalized edit distance
        edit_dist = string_metric.levenshtein(pred_text_lower_ignore,
                                              gt_text_lower_ignore)
        norm_ed = float(edit_dist) / max(1, len(gt_text_lower_ignore),
                                         len(pred_text_lower_ignore))
        norm_ed_sum += norm_ed

        # number to calculate char level recall & precision
        match_res['gt_char_num'] += len(gt_text_lower_ignore)
        match_res['pred_char_num'] += len(pred_text_lower_ignore)
        true_positive_char_num = cal_true_positive_char(
            pred_text_lower_ignore, gt_text_lower_ignore)
        match_res['true_positive_char_num'] += true_positive_char_num

    normalized_edit_distance = norm_ed_sum / max(1, len(gt_texts))
    match_res['ned'] = normalized_edit_distance

    return match_res


def eval_ocr_metric(pred_texts,
                    gt_texts,
                    metric=[
                        'word_acc', 'word_acc_ignore_case',
                        'word_acc_ignore_case_symbol', 'char_recall',
                        'char_precision', 'one_minus_ned'
                    ]):
    """Evaluate the text recognition performance with metric: word accuracy and
    1-N.E.D. See https://rrc.cvc.uab.es/?ch=14&com=tasks for details.

    Args:
        pred_texts (list[str]): Text strings of prediction.
        gt_texts (list[str]): Text strings of ground truth.
        metric (str | list[str]): Metrics to be evaluated. Options are:

          - 'word_acc': Accuracy in word level.
          - 'word_acc_ignore_case': Accuracy in word level, ignoring letter
            case.
          - 'word_acc_ignore_case_symbol': Accuracy in word level, ignoring
            letter case and symbol. (default metric for academic evaluation)
          - 'char_recall': Recall in character level, ignore
            letter case and symbol.
          - 'char_precision': Precision in character level, ignore
            letter case and symbol.
          - 'one_minus_ned': 1 - normalized_edit_distance.

    Returns:
        dict{str: float}: Result dict for text recognition, keys could be some
        of the following: ['word_acc', 'word_acc_ignore_case',
        'word_acc_ignore_case_symbol', 'char_recall', 'char_precision',
        '1-N.E.D'].
    """
    assert isinstance(pred_texts, list)
    assert isinstance(gt_texts, list)
    assert len(pred_texts) == len(gt_texts)
    assert isinstance(metric, str) or is_type_list(metric, str)
    metric = set([metric]) if isinstance(metric, str) else set(metric)

    supported_metrics = set([
        'word_acc', 'word_acc_ignore_case', 'word_acc_ignore_case_symbol',
        'char_recall', 'char_precision', 'one_minus_ned'
    ])
    assert metric.issubset(supported_metrics)

    match_res = count_matches(pred_texts, gt_texts)
    eps = 1e-8
    eval_res = {}

    if 'char_recall' in metric:
        char_recall = 1.0 * match_res['true_positive_char_num'] / (
            eps + match_res['gt_char_num'])
        eval_res['char_recall'] = char_recall

    if 'char_precision' in metric:
        char_precision = 1.0 * match_res['true_positive_char_num'] / (
            eps + match_res['pred_char_num'])
        eval_res['char_precision'] = char_precision

    if 'word_acc' in metric:
        word_acc = 1.0 * match_res['match_word_num'] / (
            eps + match_res['gt_word_num'])
        eval_res['word_acc'] = word_acc

    if 'word_acc_ignore_case' in metric:
        word_acc_ignore_case = 1.0 * match_res['match_word_ignore_case'] / (
            eps + match_res['gt_word_num'])
        eval_res['word_acc_ignore_case'] = word_acc_ignore_case

    if 'word_acc_ignore_case_symbol' in metric:
        word_acc_ignore_case_symbol = 1.0 * match_res[
            'match_word_ignore_case_symbol'] / (
                eps + match_res['gt_word_num'])
        eval_res['word_acc_ignore_case_symbol'] = word_acc_ignore_case_symbol

    if 'one_minus_ned' in metric:
        eval_res['1-N.E.D'] = 1.0 - match_res['ned']

    for key, value in eval_res.items():
        eval_res[key] = float('{:.4f}'.format(value))

    return eval_res

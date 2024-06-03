# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, Optional, Sequence, Union

from mmeval import CharRecallPrecision as _CharRecallPrecision
from mmeval import OneMinusNormEditDistance as _OneMinusNormEditDistance
from mmeval import WordAccuracy as _WordAccuracy

from mmocr.registry import METRICS


class TextRecogMixin:

    def __init__(self,
                 prefix: Optional[str] = None,
                 collect_device: Optional[str] = None) -> None:
        if prefix is not None:
            warnings.warn('DeprecationWarning: The `prefix` parameter of'
                          f' `{self.name}` is deprecated.')
        if collect_device is not None:
            warnings.warn(
                'DeprecationWarning: The `collect_device` parameter of '
                f'`{self.name}` is deprecated, use `dist_backend` instead.')

    def process(self, data_batch: Sequence[Dict],
                predictions: Sequence[Dict]) -> None:
        """Process one batch of predictions. The processed results should be
        stored in ``self.results``, which will be used to compute the metrics
        when all batches have been processed.

        Args:
            data_batch (Sequence[Dict]): A batch of gts.
            predictions (Sequence[Dict]): A batch of outputs from the model.
        """
        preds, labels = list(), list()
        for data_sample in predictions:
            preds.append(data_sample.get('pred_text').get('item'))
            labels.append(data_sample.get('gt_text').get('item'))
        self.add(preds, labels)

    def evaluate(self, size: int):
        metric_results = self.compute(size)
        metric_results = {
            f'{self.name}/{k}': v
            for k, v in metric_results.items()
        }
        self.reset()
        return metric_results


@METRICS.register_module()
class WordMetric(_WordAccuracy, TextRecogMixin):
    """Calculate the word level accuracy.

    Args:
        mode (str or list[str]): Options are:
            - 'exact': Accuracy at word level.
            - 'ignore_case': Accuracy at word level, ignoring letter
              case.
            - 'ignore_case_symbol': Accuracy at word level, ignoring
              letter case and symbol. (Default metric for academic evaluation)
            If mode is a list, then metrics in mode will be calculated
            separately. Defaults to 'ignore_case_symbol'.
        valid_symbol (str): Valid characters. Defaults to
            '[^A-Z^a-z^0-9^\u4e00-\u9fa5]'.
    """

    def __init__(self,
                 mode: Union[str, Sequence[str]] = 'ignore_case_symbol',
                 valid_symbol: str = '[^A-Z^a-z^0-9^\u4e00-\u9fa5]',
                 **kwargs) -> None:
        collect_device = kwargs.pop('collect_device', None)
        prefix = kwargs.pop('prefix', None)
        TextRecogMixin.__init__(collect_device, prefix)
        super().__init__(mode=mode, valid_symbol=valid_symbol, **kwargs)


@METRICS.register_module()
class CharMetric(_CharRecallPrecision, TextRecogMixin):
    """Calculate the char level recall & precision.

    Args:
        letter_case (str): There are three options to alter the letter cases
            - unchanged: Do not change prediction texts and labels.
            - upper: Convert prediction texts and labels into uppercase
                     characters.
            - lower: Convert prediction texts and labels into lowercase
                     characters.
            Usually, it only works for English characters. Defaults to
            'unchanged'.
        valid_symbol (str): Valid characters. Defaults to
            '[^A-Z^a-z^0-9^\u4e00-\u9fa5]'.
    """

    def __init__(self,
                 letter_case: str = 'lower',
                 valid_symbol: str = '[^A-Z^a-z^0-9^\u4e00-\u9fa5]',
                 **kwargs) -> None:
        collect_device = kwargs.pop('collect_device', None)
        prefix = kwargs.pop('prefix', None)
        super().__init__(
            letter_case=letter_case, valid_symbol=valid_symbol, **kwargs)
        TextRecogMixin.__init__(collect_device, prefix)


@METRICS.register_module()
class OneMinusNEDMetric(_OneMinusNormEditDistance, TextRecogMixin):
    """One minus NED metric for text recognition task.

    Args:
        letter_case (str): There are three options to alter the letter cases
            - unchanged: Do not change prediction texts and labels.
            - upper: Convert prediction texts and labels into uppercase
                     characters.
            - lower: Convert prediction texts and labels into lowercase
                     characters.
            Usually, it only works for English characters. Defaults to
            'unchanged'.
        valid_symbol (str): Valid characters. Defaults to
            '[^A-Z^a-z^0-9^\u4e00-\u9fa5]'.
    """

    def __init__(self,
                 letter_case: str = 'lower',
                 valid_symbol: str = '[^A-Z^a-z^0-9^\u4e00-\u9fa5]',
                 **kwargs) -> None:
        collect_device = kwargs.pop('collect_device', None)
        prefix = kwargs.pop('prefix', None)
        super().__init__(
            letter_case=letter_case, valid_symbol=valid_symbol, **kwargs)
        TextRecogMixin.__init__(collect_device, prefix)

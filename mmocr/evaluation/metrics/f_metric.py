# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, Sequence, Union

import mmeval

from mmocr.registry import METRICS


@METRICS.register_module()
class F1Metric(mmeval.F1Metric):
    """A wrapper around class:`mmeval.F1Metric`, which computes F1 scores.

    Args:
        num_classes (int): Number of labels.
        key (str): The key name of the predicted and ground truth labels.
            Defaults to 'labels'.
        mode (str or list[str]): Options are:
            - 'micro': Calculate metrics globally by counting the total true
              positives, false negatives and false positives.
            - 'macro': Calculate metrics for each label, and find their
              unweighted mean.
            If mode is a list, then metrics in mode will be calculated
            separately. Defaults to 'micro'.
        cared_classes (list[int]): The indices of the labels participated in
            the metric computing. If both ``cared_classes`` and
            ``ignored_classes`` are empty, all classes will be taken into
            account. Defaults to []. Note: ``cared_classes`` and
            ``ignored_classes`` cannot be specified together.
        ignored_classes (list[int]): The index set of labels that are ignored
            when computing metrics. If both ``cared_classes`` and
            ``ignored_classes`` are empty, all classes will be taken into
            account. Defaults to []. Note: ``cared_classes`` and
            ``ignored_classes`` cannot be specified together.
        dist_backend (str | None): The name of the distributed communication
            backend. Refer to :class:`mmeval.BaseMetric`.
            Defaults to 'torch_cuda'.
        **kwargs: Keyword parameters passed to :class:`mmeval.F1Metric`.

    Warning:
        Only non-negative integer labels are involved in computing. All
        negative ground truth labels will be ignored.
    """

    def __init__(self,
                 num_classes: int,
                 key: str = 'labels',
                 mode: Union[str, Sequence[str]] = 'micro',
                 cared_classes: Sequence[int] = [],
                 ignored_classes: Sequence[int] = [],
                 dist_backend: str = 'torch_cuda',
                 **kwargs) -> None:
        self.key = key

        prefix = kwargs.pop('prefix', None)
        if prefix is not None:
            warnings.warn('DeprecationWarning: The `prefix` parameter of'
                          ' `F1Metric` is deprecated.')

        collect_device = kwargs.pop('collect_device', None)
        if collect_device is not None:
            warnings.warn(
                'DeprecationWarning: The `collect_device` parameter of '
                '`F1Metric` is deprecated, use `dist_backend` instead.')
        super().__init__(
            num_classes,
            mode,
            cared_classes,
            ignored_classes,
            dist_backend=dist_backend,
            **kwargs)

    def process(self, data_batch: Sequence[Dict],
                data_samples: Sequence[Dict]) -> None:
        """Process one batch of data samples and predictions, and pass the
        intermidate results to ``self.add``.

        Args:
            data_batch (Sequence[Dict]): A batch of gts.
            data_samples (Sequence[Dict]): A batch of outputs from the model.
        """
        predictions = []
        labels = []
        for data_sample in data_samples:
            pred_labels = data_sample.get('pred_instances').get(self.key)
            gt_labels = data_sample.get('gt_instances').get(self.key)
            predictions.append(pred_labels)
            labels.append(gt_labels)
        self.add(predictions, labels)

    def evaluate(self, *args, **kwargs) -> Dict:
        """Compute the metrics from processed results and return the result
        with the best hmean score. All the arguments will be passed to
        ``self.compute``.

        Returns:
            dict[str, float]: The f1 scores. The keys are the names of the
                metrics, and the values are corresponding results. Possible
                keys are 'micro_f1' and 'macro_f1'.
        """

        metric_results = self.compute(*args, **kwargs)
        self.reset()
        metric_results = {
            f'{self.name}/{k}': v
            for k, v in metric_results.items()
        }
        return metric_results

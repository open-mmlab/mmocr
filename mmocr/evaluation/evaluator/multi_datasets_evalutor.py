# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict
from typing import Sequence, Union

from mmengine.dist import (broadcast_object_list, collect_results,
                           is_main_process)
from mmengine.evaluator import BaseMetric, Evaluator
from mmengine.evaluator.metric import _to_cpu

from mmocr.registry import EVALUATOR


@EVALUATOR.register_module()
class MultiDatasetsEvaluator(Evaluator):
    """Wrapper class to compose class: `ConcatDataset` and multiple
    :class:`BaseMetric` instances.
    The metrics will be evaluated on each dataset slice separately. The key of
    the metrics is the concatenation of the dataset prefix, the metric prefix
    and the key of metric - e.g. `dataset_prefix/metric_prefix/accuracy`.

    Args:
        metrics (dict or BaseMetric or Sequence): The config of metrics.
        datasets_prefix (Sequence[str]): The prefix of each dataset. The length
            of this sequence should be the same as the length of the datasets.
    """

    def __init__(self, metrics: Union[dict, BaseMetric, Sequence],
                 datasets_prefix: Sequence[str]):
        super().__init__(metrics)
        self.datasets_prefix = datasets_prefix

    def evaluate(self, size: int) -> dict:
        """Invoke ``evaluate`` method of each metric and collect the metrics
        dictionary.

        Args:
            size (int): Length of the entire validation dataset. When batch
                size > 1, the dataloader may pad some data samples to make
                sure all ranks have the same length of dataset slice. The
                ``collect_results`` function will drop the padded data based on
                this size.

        Returns:
            dict: Evaluation results of all metrics. The keys are the names
            of the metrics, and the values are corresponding results.
        """
        datasets_metrics = OrderedDict()
        dataset_slices = self.dataset_meta.get('cumulative_sizes', [size])
        assert len(dataset_slices) == len(self.datasets_prefix)
        for metric in self.metrics:
            if len(metric.results) == 0:
                warnings.warn(
                    f'{metric.__class__.__name__} got empty `self.results`.'
                    'Please ensure that the processed results are properly '
                    'added into `self.results` in `process` method.')

            results = collect_results(metric.results, size,
                                      metric.collect_device)

            if is_main_process():
                # cast all tensors in results list to cpu
                results = _to_cpu(results)
                for start, end, datasets_prefix in zip([0] +
                                                       dataset_slices[:-1],
                                                       dataset_slices,
                                                       self.datasets_prefix):
                    _metrics = metric.compute_metrics(
                        results[start:end])  # type: ignore
                    # Add prefix to metric names

                    if metric.prefix:
                        final_prefix = '/'.join(
                            (datasets_prefix, metric.prefix))
                    else:
                        final_prefix = datasets_prefix
                    _metrics = {
                        '/'.join((final_prefix, k)): v
                        for k, v in _metrics.items()
                    }

                    metric.results.clear()
                    # Check metric name conflicts
                    for name in _metrics.keys():
                        if name in datasets_metrics:
                            raise ValueError(
                                'There are multiple evaluation results with '
                                f'the same metric name {name}. Please make '
                                'sure all metrics have different prefixes.')
                    datasets_metrics.update(_metrics)
        if is_main_process():
            datasets_metrics = [datasets_metrics]
        else:
            datasets_metrics = [None]  # type: ignore
        broadcast_object_list(datasets_metrics)

        return datasets_metrics[0]

# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict
from typing import Sequence, Union

from mmengine.dist import (broadcast_object_list, collect_results,
                           is_main_process)
from mmengine.evaluator import BaseMetric, Evaluator
from mmengine.evaluator.metric import _to_cpu

from mmocr.registry import EVALUATOR
from mmocr.utils.typing_utils import ConfigType


@EVALUATOR.register_module()
class MultiDatasetsEvaluator(Evaluator):
    """Wrapper class to compose class: `ConcatDataset` and multiple
    :class:`BaseMetric` instances.
    The metrics will be evaluated on each dataset slice separately. The name of
    the each metric is the concatenation of the dataset prefix, the metric
    prefix and the key of metric - e.g.
    `dataset_prefix/metric_prefix/accuracy`.

    Args:
        metrics (dict or BaseMetric or Sequence): The config of metrics.
        dataset_prefixes (Sequence[str]): The prefix of each dataset. The
            length of this sequence should be the same as the length of the
            datasets.
    """

    def __init__(self, metrics: Union[ConfigType, BaseMetric, Sequence],
                 dataset_prefixes: Sequence[str]) -> None:
        super().__init__(metrics)
        self.dataset_prefixes = dataset_prefixes

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
        metrics_results = OrderedDict()
        dataset_slices = self.dataset_meta.get('cumulative_sizes', [size])
        assert len(dataset_slices) == len(self.dataset_prefixes)
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
                for start, end, dataset_prefix in zip([0] +
                                                      dataset_slices[:-1],
                                                      dataset_slices,
                                                      self.dataset_prefixes):
                    metric_results = metric.compute_metrics(
                        results[start:end])  # type: ignore
                    # Add prefix to metric names

                    if metric.prefix:
                        final_prefix = '/'.join(
                            (dataset_prefix, metric.prefix))
                    else:
                        final_prefix = dataset_prefix
                    metric_results = {
                        '/'.join((final_prefix, k)): v
                        for k, v in metric_results.items()
                    }

                    # Check metric name conflicts
                    for name in metric_results.keys():
                        if name in metrics_results:
                            raise ValueError(
                                'There are multiple evaluation results with '
                                f'the same metric name {name}. Please make '
                                'sure all metrics have different prefixes.')
                    metrics_results.update(metric_results)
            metric.results.clear()
        if is_main_process():
            averaged_results = [self.average_results(metrics_results)]
        else:
            averaged_results = [None]

        metrics_results = [metrics_results]
        broadcast_object_list(metrics_results)
        broadcast_object_list([averaged_results])
        results = {
            'metric_results': metrics_results[0],
            'averaged_results': averaged_results
        }
        return results

    def average_results(self, metrics_results):
        """Compute the average of metric results across all datasets.

        Args:
            metrics_results (dict): Evaluation results of all metrics.

        Returns:pre
            dict: Average evaluation results of all metrics.
        """
        averaged_results = {}
        num_datasets = len(self.dataset_prefixes)
        for metric_name, metric_result in metrics_results.items():
            metric_avg = metric_result / num_datasets
            averaged_results[metric_name] = metric_avg

        return averaged_results

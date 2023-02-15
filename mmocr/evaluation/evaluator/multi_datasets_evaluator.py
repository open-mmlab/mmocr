# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict
from typing import Sequence, Union

from mmengine.dist import broadcast_object_list, is_main_process
from mmengine.evaluator import BaseMetric, Evaluator

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
            if len(metric._results) == 0:
                warnings.warn(
                    f'{metric.__class__.__name__} got empty `self.results`.'
                    'Please ensure that the processed results are properly '
                    'added into `self.results` in `process` method.')

            global_results = metric.dist_comm.all_gather_object(
                metric._results)
            if metric.dist_collect_mode == 'cat':
                # use `sum` to concatenate list
                # e.g. sum([[1, 3], [2, 4]], []) = [1, 3, 2, 4]
                collected_results = sum(global_results, [])
            else:
                collected_results = []
                for partial_result in zip(*global_results):
                    collected_results.extend(list(partial_result))
            if is_main_process():
                for start, end, dataset_prefix in zip([0] +
                                                      dataset_slices[:-1],
                                                      dataset_slices,
                                                      self.dataset_prefixes):
                    metric_results = metric.compute_metric(
                        collected_results[start:end])  # type: ignore
                    # Add prefix to metric names

                    final_prefix = '/'.join((dataset_prefix, metric.name))
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
            metric.reset()
        if is_main_process():
            metrics_results = [metrics_results]
        else:
            metrics_results = [None]  # type: ignore
        broadcast_object_list(metrics_results)

        return metrics_results[0]

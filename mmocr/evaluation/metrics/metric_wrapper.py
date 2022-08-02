# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import List, Sequence

from mmengine.dist import (broadcast_object_list, collect_results,
                           is_main_process)
from mmengine.evaluator import BaseMetric
from mmengine.evaluator.metric import _to_cpu

from mmocr.registry import METRICS


@METRICS.register_module()
class MultiDatasetMetricWrapper(BaseMetric):
    """multi dataset metric wrapper.

    Args:
        metric (BaseMetric): metric to be wrapped
        dataset_slices (list): list of dataset slices
        averge (bool): whether to average the metric across the dataset
    """

    def __init__(self, metric, prefix: List) -> None:
        super().__init__()
        self.metric = METRICS.build(metric)
        self.prefix = prefix

    def evaluate(self, size: int) -> dict:
        """Evaluate the model performance of the whole dataset after processing
        all batches.

        Args:
            size (int): Length of the entire validation dataset. When batch
                size > 1, the dataloader may pad some data samples to make
                sure all ranks have the same length of dataset slice. The
                ``collect_results`` function will drop the padded data based on
                this size.

        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are the
            names of the metrics, and the values are corresponding results.
        """
        if len(self.metric.results) == 0:
            warnings.warn(
                f'{self.metric.__class__.__name__} got empty `self.results`.'
                'Please ensure that the processed results are properly added'
                ' into `self.results` in `process` method.')

        results = collect_results(self.metric.results, size,
                                  self.collect_device)

        if is_main_process():
            # cast all tensors in results list to cpu
            results = _to_cpu(results)
            dataset_metrics = dict()
            dataset_slices = self.dataset_meta.get('cumulative_sizes', [size])
            for start, end, prefix in zip([0] + dataset_slices[:-1],
                                          dataset_slices, self.prefix):
                _metrics = self.compute_metrics(results[start:end])
                _metrics = {
                    '/'.join((prefix, k)): v
                    for k, v in _metrics.items()
                }
                dataset_metrics.update(_metrics)
            metrics = [dataset_metrics]
        else:
            metrics = [None]  # type: ignore

        broadcast_object_list(metrics)

        # reset the results list
        self.metric.results.clear()
        return metrics[0]

    def process(self, data_batch: Sequence[dict],
                predictions: Sequence[dict]) -> None:
        self.metric.process(data_batch, predictions)

    def compute_metrics(self, results: list) -> dict:
        return self.metric.compute_metrics(results)

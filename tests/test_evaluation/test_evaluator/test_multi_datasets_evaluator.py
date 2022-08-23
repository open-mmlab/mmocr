# Copyright (c) OpenMMLab. All rights reserved.

import math
from typing import Dict, List, Optional
from unittest import TestCase

import numpy as np
from mmengine import METRICS, BaseDataElement, DefaultScope
from mmengine.evaluator import BaseMetric

from mmocr.evaluation import MultiDatasetsEvaluator


@METRICS.register_module()
class ToyMetric(BaseMetric):
    """Evaluator that calculates the metric `accuracy` from predictions and
    labels. Alternatively, this evaluator can return arbitrary dummy metrics
    set in the config.

    Default prefix: Toy

    Metrics:
        - accuracy (float): The classification accuracy. Only when
            `dummy_metrics` is None.
        - size (int): The number of test samples. Only when `dummy_metrics`
            is None.

        If `dummy_metrics` is set as a dict in the config, it will be
        returned as the metrics and override `accuracy` and `size`.
    """

    default_prefix = None

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = 'Toy',
                 dummy_metrics: Optional[Dict] = None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.dummy_metrics = dummy_metrics

    def process(self, data_batch, predictions):
        results = [{
            'pred': pred.get('pred'),
            'label': data['data_sample'].get('label')
        } for pred, data in zip(predictions, data_batch)]
        self.results.extend(results)

    def compute_metrics(self, results: List):
        if self.dummy_metrics is not None:
            assert isinstance(self.dummy_metrics, dict)
            return self.dummy_metrics.copy()

        pred = np.array([result['pred'] for result in results])
        label = np.array([result['label'] for result in results])
        acc = (pred == label).sum() / pred.size

        metrics = {
            'accuracy': acc,
            'size': pred.size,  # To check the number of testing samples
        }

        return metrics


def generate_test_results(size, batch_size, pred, label):
    num_batch = math.ceil(size / batch_size)
    bs_residual = size % batch_size
    for i in range(num_batch):
        bs = bs_residual if i == num_batch - 1 else batch_size
        data_batch = [
            dict(
                inputs=np.zeros((3, 10, 10)),
                data_sample=BaseDataElement(label=label)) for _ in range(bs)
        ]
        predictions = [BaseDataElement(pred=pred) for _ in range(bs)]
        yield (data_batch, predictions)


class TestMultiDatasetsEvaluator(TestCase):

    def test_composed_metrics(self):
        DefaultScope.get_instance('mmocr_metric', scope_name='mmocr')
        cfg = [
            dict(type='ToyMetric'),
            dict(type='ToyMetric', dummy_metrics=dict(mAP=0.0))
        ]

        evaluator = MultiDatasetsEvaluator(cfg, dataset_prefixes=['Fake'])
        evaluator.dataset_meta = {}
        size = 10
        batch_size = 4

        for data_samples, predictions in generate_test_results(
                size, batch_size, pred=1, label=1):
            evaluator.process(data_samples, predictions)

        metrics = evaluator.evaluate(size=size)

        self.assertAlmostEqual(metrics['Fake/Toy/accuracy'], 1.0)
        self.assertAlmostEqual(metrics['Fake/Toy/mAP'], 0.0)
        self.assertEqual(metrics['Fake/Toy/size'], size)
        with self.assertWarns(Warning):
            evaluator.evaluate(size=0)

        cfg = [dict(type='ToyMetric'), dict(type='ToyMetric')]

        evaluator = MultiDatasetsEvaluator(cfg, dataset_prefixes=['Fake'])
        evaluator.dataset_meta = {}

        for data_samples, predictions in generate_test_results(
                size, batch_size, pred=1, label=1):
            evaluator.process(data_samples, predictions)
        with self.assertRaises(ValueError):
            evaluator.evaluate(size=size)

        cfg = [dict(type='ToyMetric'), dict(type='ToyMetric', prefix=None)]

        evaluator = MultiDatasetsEvaluator(cfg, dataset_prefixes=['Fake'])
        evaluator.dataset_meta = {}

        for data_samples, predictions in generate_test_results(
                size, batch_size, pred=1, label=1):
            evaluator.process(data_samples, predictions)
        metrics = evaluator.evaluate(size=size)
        self.assertIn('Fake/Toy/accuracy', metrics)
        self.assertIn('Fake/accuracy', metrics)

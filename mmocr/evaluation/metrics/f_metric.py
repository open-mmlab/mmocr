# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence, Union

import torch
from mmengine.evaluator import BaseMetric

from mmocr.registry import METRICS


@METRICS.register_module()
class F1Metric(BaseMetric):
    """Compute F1 scores.

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
        cared_classes (list[int]): The indices of the labels particpated in
            the metirc computing. If both ``cared_classes`` and
            ``ignored_classes`` are empty, all classes will be taken into
            account. Defaults to []. Note: ``cared_classes`` and
            ``ignored_classes`` cannot be specified together.
        ignored_classes (list[int]): The index set of labels that are ignored
            when computing metrics. If both ``cared_classes`` and
            ``ignored_classes`` are empty, all classes will be taken into
            account. Defaults to []. Note: ``cared_classes`` and
            ``ignored_classes`` cannot be specified together.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.

    Warning:
        Only non-negative integer labels are involved in computing. All
        negative ground truth labels will be ignored.
    """

    default_prefix: Optional[str] = 'kie'

    def __init__(self,
                 num_classes: int,
                 key: str = 'labels',
                 mode: Union[str, Sequence[str]] = 'micro',
                 cared_classes: Sequence[int] = [],
                 ignored_classes: Sequence[int] = [],
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device, prefix)
        assert isinstance(num_classes, int)
        assert isinstance(cared_classes, (list, tuple))
        assert isinstance(ignored_classes, (list, tuple))
        assert isinstance(mode, (list, str))
        assert not (len(cared_classes) > 0 and len(ignored_classes) > 0), \
            'cared_classes and ignored_classes cannot be both non-empty'

        if isinstance(mode, str):
            mode = [mode]
        assert set(mode).issubset({'micro', 'macro'})
        self.mode = mode

        if len(cared_classes) > 0:
            assert min(cared_classes) >= 0 and \
                max(cared_classes) < num_classes, \
                'cared_classes must be a subset of [0, num_classes)'
            self.cared_labels = sorted(cared_classes)
        elif len(ignored_classes) > 0:
            assert min(ignored_classes) >= 0 and \
                max(ignored_classes) < num_classes, \
                'ignored_classes must be a subset of [0, num_classes)'
            self.cared_labels = sorted(
                set(range(num_classes)) - set(ignored_classes))
        else:
            self.cared_labels = list(range(num_classes))
        self.num_classes = num_classes
        self.key = key

    def process(self, data_batch: Sequence[Dict],
                data_samples: Sequence[Dict]) -> None:
        """Process one batch of data_samples. The processed results should be
        stored in ``self.results``, which will be used to compute the metrics
        when all batches have been processed.

        Args:
            data_batch (Sequence[Dict]): A batch of gts.
            data_samples (Sequence[Dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            pred_labels = data_sample.get('pred_instances').get(self.key).cpu()
            gt_labels = data_sample.get('gt_instances').get(self.key).cpu()

            result = dict(
                pred_labels=pred_labels.flatten(),
                gt_labels=gt_labels.flatten())
            self.results.append(result)

    def compute_metrics(self, results: Sequence[Dict]) -> Dict:
        """Compute the metrics from processed results.

        Args:
            results (list[Dict]): The processed results of each batch.

        Returns:
            dict[str, float]: The f1 scores. The keys are the names of the
                metrics, and the values are corresponding results. Possible
                keys are 'micro_f1' and 'macro_f1'.
        """

        preds = []
        gts = []
        for result in results:
            preds.append(result['pred_labels'])
            gts.append(result['gt_labels'])
        preds = torch.cat(preds)
        gts = torch.cat(gts)

        assert preds.max() < self.num_classes
        assert gts.max() < self.num_classes

        cared_labels = preds.new_tensor(self.cared_labels, dtype=torch.long)

        hits = (preds == gts)[None, :]
        preds_per_label = cared_labels[:, None] == preds[None, :]
        gts_per_label = cared_labels[:, None] == gts[None, :]

        tp = (hits * preds_per_label).float()
        fp = (~hits * preds_per_label).float()
        fn = (~hits * gts_per_label).float()

        result = {}
        if 'macro' in self.mode:
            result['macro_f1'] = self._compute_f1(
                tp.sum(-1), fp.sum(-1), fn.sum(-1))
        if 'micro' in self.mode:
            result['micro_f1'] = self._compute_f1(tp.sum(), fp.sum(), fn.sum())

        return result

    def _compute_f1(self, tp: torch.Tensor, fp: torch.Tensor,
                    fn: torch.Tensor) -> float:
        """Compute the F1-score based on the true positives, false positives
        and false negatives.

        Args:
            tp (Tensor): The true positives.
            fp (Tensor): The false positives.
            fn (Tensor): The false negatives.

        Returns:
            float: The F1-score.
        """
        precision = tp / (tp + fp).clamp(min=1e-8)
        recall = tp / (tp + fn).clamp(min=1e-8)
        f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
        return float(f1.mean())

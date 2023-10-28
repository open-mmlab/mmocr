from typing import Any, Optional, Sequence

from mmengine.evaluator import BaseMetric
from seqeval.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

from mmocr.registry import METRICS


@METRICS.register_module()
class SeqevalMetric(BaseMetric):

    default_prefix: Optional[str] = 'ser'

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device, prefix)

    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            pred_labels = data_sample.get('pred_label').get('item')
            gt_labels = data_sample.get('gt_label').get('item')

            result = dict(pred_labels=pred_labels, gt_labels=gt_labels)
            self.results.append(result)

    def compute_metrics(self, results: list) -> dict:
        preds = []
        gts = []
        for result in results:
            preds.append(result['pred_labels'])
            gts.append(result['gt_labels'])

        result = {
            'precision': precision_score(gts, preds),
            'recall': recall_score(gts, preds),
            'f1': f1_score(gts, preds),
            'accuracy': accuracy_score(gts, preds)
        }
        return result

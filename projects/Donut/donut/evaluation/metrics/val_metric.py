from typing import Optional, Sequence, Dict
import re
import numpy as np
from nltk import edit_distance
from mmengine.evaluator import BaseMetric

from mmocr.registry import METRICS


@METRICS.register_module()
class DonutValEvaluator(BaseMetric):
    default_prefix: Optional[str] = ''
    def __init__(self,
                 key: str = 'parses',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device, prefix)
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
            pred_parses = data_sample.get('pred_instances').get(self.key)[0]
            gt_parses = data_sample.get('gt_instances').get(self.key)[0]

            result = dict(
                pred_labels=pred_parses,
                gt_labels=gt_parses)
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

        scores = []
        for result in results:
            pred = result['pred_labels']
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            gt = result['gt_labels']
            scores.append(edit_distance(pred, gt) / max(len(pred), len(gt)))

        val_metric = np.mean(scores)

        result = {}
        result['val_metric'] = val_metric
        return result

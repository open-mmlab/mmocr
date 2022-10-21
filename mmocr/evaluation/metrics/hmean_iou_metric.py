# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, Sequence

import mmeval
import torch
from mmengine.logging import MMLogger

from mmocr.registry import METRICS


@METRICS.register_module()
class HmeanIOUMetric(mmeval.HmeanIoU):
    """A wrapper around class:`mmeval.HmeanIoU`.

    This method computes the hmean iou metric, which is done in the
    following steps:

    - Filter the prediction polygon:

      - Scores is smaller than minimum prediction score threshold.
      - The proportion of the area that intersects with gt ignored polygon is
        greater than ignore_precision_thr.

    - Computing an M x N IoU matrix, where each element indexing
      E_mn represents the IoU between the m-th valid GT and n-th valid
      prediction.
    - Based on different prediction score threshold:
      - Obtain the ignored predictions according to prediction score.
        The filtered predictions will not be involved in the later metric
        computations.
      - Based on the IoU matrix, get the match metric according to
      ``match_iou_thr``.
      - Based on different `strategy`, accumulate the match number.
    - calculate H-mean under different prediction score threshold.

    Args:
        match_iou_thr (float): IoU threshold for a match. Defaults to 0.5.
        ignore_precision_thr (float): Precision threshold when prediction and\
            gt ignored polygons are matched. Defaults to 0.5.
        pred_score_thrs (dict): Best prediction score threshold searching
            space. Defaults to dict(start=0.3, stop=0.9, step=0.1).
        strategy (str): Polygon matching strategy. Options are 'max_matching'
            and 'vanilla'. 'max_matching' refers to the optimum strategy that
            maximizes the number of matches. Vanilla strategy matches gt and
            pred polygons if both of them are never matched before. It was used
            in MMOCR 0.x and and academia. Defaults to 'vanilla'.
        dist_backend (str | None): The name of the distributed communication
            backend. Refer to :class:`mmeval.BaseMetric`.
            Defaults to 'torch_cuda'.
        **kwargs: Keyword parameters passed to :class:`mmeval.HmeanIoU`.
    """

    def __init__(self,
                 match_iou_thr: float = 0.5,
                 ignore_precision_thr: float = 0.5,
                 pred_score_thrs: Dict = dict(start=0.3, stop=0.9, step=0.1),
                 strategy: str = 'vanilla',
                 dist_backend: str = 'torch_cuda',
                 **kwargs) -> None:
        prefix = kwargs.pop('prefix', None)
        if prefix is not None:
            warnings.warn('DeprecationWarning: The `prefix` parameter of'
                          ' `HmeanIOUMetric` is deprecated.')

        collect_device = kwargs.pop('collect_device', None)
        if collect_device is not None:
            warnings.warn(
                'DeprecationWarning: The `collect_device` parameter of '
                '`HmeanIOUMetric` is deprecated, use `dist_backend` instead.')

        super().__init__(
            match_iou_thr,
            ignore_precision_thr,
            pred_score_thrs,
            strategy,
            dist_backend=dist_backend,
            **kwargs)

    def process(self, data_batch: Sequence[Dict],
                data_samples: Sequence[Dict]) -> None:
        """Process one batch of data samples and predictions, and pass the
        intermidate results to ``self.add``.

        Args:
            data_batch (Sequence[Dict]): A batch of data from dataloader.
            data_samples (Sequence[Dict]): A batch of outputs from
                the model.
        """
        batch_pred_polygons = []
        batch_pred_scores = []
        batch_gt_polygons = []
        batch_gt_ignore_flags = []
        for data_sample in data_samples:
            pred_instances = data_sample.get('pred_instances')
            gt_instances = data_sample.get('gt_instances')

            batch_pred_polygons.append(pred_instances.get('polygons'))
            batch_gt_polygons.append(gt_instances.get('polygons'))

            pred_scores = pred_instances.get('scores')
            if isinstance(pred_scores, torch.Tensor):
                pred_scores = pred_scores.cpu().numpy()
            batch_pred_scores.append(pred_scores)

            gt_ignore_flags = gt_instances.get('ignored')
            if isinstance(gt_ignore_flags, torch.Tensor):
                gt_ignore_flags = gt_ignore_flags.cpu().numpy()
            batch_gt_ignore_flags.append(gt_ignore_flags)
        self.add(batch_pred_polygons, batch_pred_scores, batch_gt_polygons,
                 batch_gt_ignore_flags)

    def evaluate(self, *args, **kwargs) -> Dict:
        """Compute the metrics from processed results and return the result
        with the best hmean score. All the arguments will be passed to
        ``self.compute``.

        Returns:
            dict[str, float]: The metric results with the best hmean score. The
            keys are "precision", "recall" and "hmean".
        """
        logger: MMLogger = MMLogger.get_current_instance()

        logger.info('Evaluating hmean-iou...')
        metric_results = self.compute(*args, **kwargs)
        self.reset()
        best_eval_results = metric_results.pop('best')

        for pred_score_thr in sorted(list(metric_results.keys())):
            result = metric_results[pred_score_thr]
            logger.info(f'prediction score threshold: {pred_score_thr:.2f}, '
                        f'recall: {result["recall"]:.4f}, '
                        f'precision: {result["precision"]:.4f}, '
                        f'hmean: {result["hmean"]:.4f}\n')

        return best_eval_results

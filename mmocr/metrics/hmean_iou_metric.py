# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from shapely.geometry import Polygon

from mmocr.core.evaluation.utils import compute_hmean
from mmocr.registry import METRICS
from mmocr.utils import poly_intersection, poly_iou, polys2shapely


@METRICS.register_module()
class HmeanIOUMetric(BaseMetric):
    """HmeanIOU metric.

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
            in MMOCR 0.x and is not recommended to use now. Defaults to
            'max_matching'.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None
    """
    default_prefix: Optional[str] = 'icdar'

    def __init__(self,
                 match_iou_thr: float = 0.5,
                 ignore_precision_thr: float = 0.5,
                 pred_score_thrs: Dict = dict(start=0.3, stop=0.9, step=0.1),
                 strategy: str = 'max_matching',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.match_iou_thr = match_iou_thr
        self.ignore_precision_thr = ignore_precision_thr
        self.pred_score_thrs = np.arange(**pred_score_thrs)
        assert strategy in ['max_matching', 'vanilla']
        self.strategy = strategy

    def process(self, data_batch: Sequence[Dict],
                predictions: Sequence[Dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[Dict]): A batch of data from dataloader.
            predictions (Sequence[Dict]): A batch of outputs from
                the model.
        """
        for pred, gt in zip(predictions, data_batch):

            pred_instances = pred.get('pred_instances')
            pred_polygons = pred_instances.get('polygons')
            pred_scores = pred_instances.get('scores')
            if isinstance(pred_scores, torch.Tensor):
                pred_scores = pred_scores.cpu().numpy()
            pred_scores = np.array(pred_scores, dtype=np.float32)

            gt_polys, gt_ignore_flags = self._polys_from_ann(
                gt['data_sample']['instances'])
            gt_polys = polys2shapely(gt_polys)
            pred_polys = polys2shapely(pred_polygons)

            pred_ignore_flags = self._filter_preds(pred_polys, gt_polys,
                                                   pred_scores,
                                                   gt_ignore_flags)

            gt_num = np.sum(~gt_ignore_flags)
            pred_num = np.sum(~pred_ignore_flags)
            iou_metric = np.zeros([gt_num, pred_num])

            # Compute IoU scores amongst kept pred and gt polygons
            for pred_mat_id, pred_poly_id in enumerate(
                    self._true_indexes(~pred_ignore_flags)):
                for gt_mat_id, gt_poly_id in enumerate(
                        self._true_indexes(~gt_ignore_flags)):
                    iou_metric[gt_mat_id, pred_mat_id] = poly_iou(
                        gt_polys[gt_poly_id], pred_polys[pred_poly_id])

            result = dict(
                iou_metric=iou_metric,
                pred_scores=pred_scores[~pred_ignore_flags])
            self.results.append(result)

    def compute_metrics(self, results: List[Dict]) -> Dict:
        """Compute the metrics from processed results.

        Args:
            results (list[dict]): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        best_eval_results = dict(hmean=-1)
        logger.info('Evaluating hmean-iou...')

        dataset_pred_num = np.zeros_like(self.pred_score_thrs)
        dataset_hit_num = np.zeros_like(self.pred_score_thrs)
        dataset_gt_num = 0

        for result in results:
            iou_metric = result['iou_metric']  # (gt_num, pred_num)
            pred_scores = result['pred_scores']  # (pred_num)
            dataset_gt_num += iou_metric.shape[0]

            # Filter out predictions by IoU threshold
            for i, pred_score_thr in enumerate(self.pred_score_thrs):
                pred_ignore_flags = pred_scores < pred_score_thr
                # get the number of matched boxes
                matched_metric = iou_metric[:, ~pred_ignore_flags] \
                    > self.match_iou_thr
                if self.strategy == 'max_matching':
                    csr_matched_metric = csr_matrix(matched_metric)
                    matched_preds = maximum_bipartite_matching(
                        csr_matched_metric, perm_type='row')
                    # -1 denotes unmatched pred polygons
                    dataset_hit_num[i] += np.sum(matched_preds != -1)
                else:
                    # first come first matched
                    matched_gt_indexes = set()
                    matched_pred_indexes = set()
                    for gt_idx, pred_idx in zip(*np.nonzero(matched_metric)):
                        if gt_idx in matched_gt_indexes or \
                          pred_idx in matched_pred_indexes:
                            continue
                        matched_gt_indexes.add(gt_idx)
                        matched_pred_indexes.add(pred_idx)
                    dataset_hit_num[i] += len(matched_gt_indexes)
                dataset_pred_num[i] += np.sum(~pred_ignore_flags)

        for i, pred_score_thr in enumerate(self.pred_score_thrs):
            precision, recall, hmean = compute_hmean(
                int(dataset_hit_num[i]), int(dataset_hit_num[i]),
                int(dataset_gt_num), int(dataset_pred_num[i]))
            eval_results = dict(
                precision=precision, recall=recall, hmean=hmean)
            logger.info(f'prediction score threshold: {pred_score_thr}, '
                        f'recall: {eval_results["recall"]:.3f}, '
                        f'precision: {eval_results["precision"]:.3f}, '
                        f'hmean: {eval_results["hmean"]:.3f}\n')
            if eval_results['hmean'] > best_eval_results['hmean']:
                best_eval_results = eval_results
        return best_eval_results

    def _filter_preds(self, pred_polys: List[Polygon], gt_polys: List[Polygon],
                      pred_scores: List[float],
                      gt_ignore_flags: np.ndarray) -> np.ndarray:
        """Filter out the predictions by score threshold and whether it
        overlaps ignored gt polygons.

        Args:
            pred_polys (list[Polygon]): Pred polygons.
            gt_polys (list[Polygon]): GT polygons.
            pred_scores (list[float]): Pred scores of polygons.
            gt_ignore_flags (np.ndarray): 1D boolean array indicating
                the positions of ignored gt polygons.

        Returns:
            np.ndarray: 1D boolean array indicating the positions of ignored
            pred polygons.
        """

        # Filter out predictions based on the minimum score threshold
        pred_ignore_flags = pred_scores < self.pred_score_thrs.min()

        # Filter out pred polygons which overlaps any ignored gt polygons
        for pred_id in self._true_indexes(~pred_ignore_flags):
            for gt_id in self._true_indexes(gt_ignore_flags):
                # Match pred with ignored gt
                precision = poly_intersection(
                    gt_polys[gt_id], pred_polys[pred_id]) / (
                        pred_polys[pred_id].area + 1e-5)
                if precision > self.ignore_precision_thr:
                    pred_ignore_flags[pred_id] = True
                    break

        return pred_ignore_flags

    def _true_indexes(self, array: np.ndarray) -> np.ndarray:
        """Get indexes of True elements from a 1D boolean array."""
        return np.where(array)[0]

    def _polys_from_ann(self, ann: Dict) -> Tuple[List, List]:
        """Get GT polygons from annotations.

        Args:
            ann (dict): The ground-truth annotation.

        Returns:
            tuple[list[np.array], np.array]: Returns a tuple
            ``(polys, gt_ignore_flags)``, where ``polys`` is the ground-truth
            polygon instances and ``gt_ignore_flags`` represents whether the
            corresponding instance should be ignored.
        """
        polys = []
        gt_ignore_flags = []
        for instance in ann:
            gt_ignore_flags.append(instance['ignore'])
            polys.append(np.array(instance['polygon'], dtype=np.float32))
        return polys, np.array(gt_ignore_flags, dtype=bool)

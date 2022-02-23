# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod


class BaseAssigner(metaclass=ABCMeta):
    """Base assigner that assigns boxes to ground truth boxes."""

    def __init__(self, assign_key='bboxes'):
        self.assign_key = assign_key

    @abstractmethod
    def _assign(self,
                bboxes,
                gt_bboxes,
                gt_bboxes_ignore=None,
                gt_labels=None):
        """Assign boxes to ground truth boxes."""

    def assign(self, pred_results, gt_results):
        gt_inds = self._assign(pred_results[self.assign_key],
                               gt_results['gt_' + self.assign_key])
        return gt_inds

# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.structures import InstanceData

from mmocr.registry import TASK_UTILS


@TASK_UTILS.register_module()
class ConcatSampler:

    def sample(self, gt_instances, pred_instances):
        if len(pred_instances) == 0:
            return gt_instances
        proposals = InstanceData()
        proposals.texts = gt_instances.texts + gt_instances[
            pred_instances.assign_index].texts
        proposals.beziers = torch.cat(
            [gt_instances.beziers, pred_instances.beziers], dim=0)
        return proposals


@TASK_UTILS.register_module()
class OnlyGTSampler:

    def sample(self, gt_instances, pred_instances):
        return gt_instances[~gt_instances.ignored]

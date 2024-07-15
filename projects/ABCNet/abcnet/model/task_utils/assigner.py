# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmocr.registry import TASK_UTILS


@TASK_UTILS.register_module()
class L1DistanceAssigner:

    def assign(self, gt_instances, pred_instances):
        gt_beziers = gt_instances.beziers
        pred_beziers = pred_instances.beziers
        assign_index = [
            int(
                torch.argmin(
                    torch.abs(gt_beziers - pred_beziers[i]).sum(dim=1)))
            for i in range(len(pred_beziers))
        ]
        pred_instances.assign_index = assign_index
        return gt_instances, pred_instances

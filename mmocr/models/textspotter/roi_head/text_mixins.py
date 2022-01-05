# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
import torch
from mmdet.core import bbox2roi


class RecTestMixin:

    def simple_test_rec(self, x, img_metas, proposal_list, det_bboxes,
                        det_labels, rescale, **kwargs):
        """Simple test for rec head without augmentation."""
        # image shapes of images in the batch

        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        if isinstance(scale_factors[0], float):
            warnings.warn(
                'Scale factor in img_metas should be a '
                'ndarray with shape (4,) '
                'arrange as (factor_w, factor_h, factor_w, factor_h), '
                'The scale_factor with float type has been deprecated. ')
            scale_factors = np.array([scale_factors] * 4, dtype=np.float32)

        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            recognition_results = []
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale:
                scale_factors = [
                    torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                    for scale_factor in scale_factors
                ]
            _bboxes = [
                det_bboxes[i][:, :] *
                scale_factors[i] if rescale else det_bboxes[i][:, :]
                for i in range(len(det_bboxes))
            ]
            recognition_rois = bbox2roi(_bboxes)
            recognition_results = self._recognition_forward(
                x, recognition_rois)
            recognition_pred = recognition_results['recognition_pred']
            # split batch recognition prediction back to each image
            num_recognition_roi_per_img = [
                len(det_bbox) for det_bbox in det_bboxes
            ]
            recognition_pred = recognition_pred.split(
                num_recognition_roi_per_img, 0)
            proposal_list = proposal_list.split(num_recognition_roi_per_img, 0)
            # apply post-processing to each image individually
            recognition_results = self.get_results(det_bboxes, det_labels,
                                                   proposal_list,
                                                   recognition_pred)
        return recognition_results

    def get_results(self, **kwargs):
        pass

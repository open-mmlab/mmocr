# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial

import numpy as np
import torch.nn as nn

from mmocr.models.builder import build_postprocessor


class BaseTwoStageTextSpotterPostProcessor(nn.Module):

    def __init__(self,
                 det_postprocessor=None,
                 recog_postprocessor=None,
                 train_cfg=None,
                 test_cfg=None):
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.det_postprocessor = None
        self.recog_postprocessor = None
        if det_postprocessor:
            det_train_cfg = train_cfg.pop('det_postprocessor', None)
            det_test_cfg = test_cfg.pop('det_postprocessor', None)

            det_postprocessor.update(
                dict(train_cfg=det_train_cfg, test_cfg=det_test_cfg))
            self.det_postprocessor = build_postprocessor(det_postprocessor)

        if recog_postprocessor:
            self.recog_postprocessor = build_postprocessor(recog_postprocessor)

    def forward(self,
                det_pred_results,
                recog_pred_results,
                img_metas=None,
                **kwargs):
        cfg = self.train_cfg if self.training else self.test_cfg
        cfg.update(kwargs)
        if self.det_postprocessor is not None:
            det_results = self.det_postprocessor(
                det_pred_results=det_pred_results,
                img_metas=img_metas,
                **kwargs)
        else:
            assert isinstance(det_pred_results, list)
            det_results = det_pred_results
        if self.recog_postprocessor is not None:
            recog_results = self.recog_postprocessor(recog_pred_results)
        else:
            assert isinstance(recog_pred_results, list)
            recog_results = recog_pred_results

        if len(img_metas) > 1:
            scale_factors = [meta[0]['scale_factor'] for meta in img_metas]
        else:
            scale_factors = [img_metas[0]['scale_factor']]
        forward_single = partial(self._forward_single, **cfg)
        results = list(
            map(forward_single, det_results, recog_results, scale_factors))

        return results

    def rescale_results(self, results, scale_factor, rescale_fields=None):
        """Rescale results via scale_factor."""
        assert isinstance(scale_factor, np.ndarray)
        assert scale_factor.shape[0] == 4
        _rescale_single_result = partial(
            self._rescale_single_result, scale_factor=scale_factor)
        for key in rescale_fields:
            results[key] = list(map(_rescale_single_result, results[key]))
        return results

    def _rescale_single_result(self, polygon, scale_factor):
        polygon = np.array(polygon)
        poly_shape = polygon.shape
        reshape_polygon = polygon.reshape(1, -1)
        single_instance_point_num = reshape_polygon.shape[-1] / 2
        scale_factor = np.repeat(scale_factor[:2], single_instance_point_num)
        polygon = (reshape_polygon * scale_factor).reshape(poly_shape).tolist()
        return polygon

    def merge_text_spotter_result(self, det_result, recog_result):
        results = dict()
        results.update(det_result)
        results.update(recog_result)
        return results

    def _forward_single(self,
                        det_pred_result,
                        recog_pred_result,
                        scale_factor=None,
                        rescale=True,
                        rescale_fields=[],
                        **kwargs):

        if rescale and rescale_fields:
            det_pred_result = self.rescale_results(det_pred_result,
                                                   scale_factor,
                                                   rescale_fields)

        results = self.merge_text_spotter_result(det_pred_result,
                                                 recog_pred_result)
        return results

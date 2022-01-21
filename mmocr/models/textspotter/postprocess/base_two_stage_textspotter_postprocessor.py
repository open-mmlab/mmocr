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
        if det_postprocessor:
            det_train_cfg = train_cfg.pop('det_postprocessor', None)
            det_test_cfg = test_cfg.pop('det_postprocessor', None)

            det_postprocessor.update(
                dict(train_cfg=det_train_cfg, test_cfg=det_test_cfg))
            self.det_postprocessor = build_postprocessor(det_postprocessor)
        else:
            self.det_postprocessor = None

        if recog_postprocessor:
            self.recog_postprocessor = build_postprocessor(recog_postprocessor)
        else:
            self.recog_postprocessor = None

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
            assert type(det_pred_results) is list
            det_results = det_pred_results
        if self.recog_postprocessor is not None:
            recog_results = self.recog_postprocessor(recog_pred_results)
        else:
            assert type(recog_pred_results) is list
            recog_results = recog_pred_results

        if len(img_metas) > 1:
            scale_factors = [meta[0]['scale_factor'] for meta in img_metas]
        else:
            scale_factors = [img_metas[0]['scale_factor']]
        forward_single = partial(self._forward_single, **cfg)
        results = list(
            map(forward_single, det_results, recog_results, scale_factors))

        return results

    def rescale_results(self, results, scale_factor, property=None):
        """Rescale results via scale_factor."""
        assert isinstance(scale_factor, np.ndarray)
        assert scale_factor.shape[0] == 4
        for key in property:
            _rescale_single_result = partial(
                self._rescale_single_result, scale_factor=scale_factor)
            results[key] = list(map(_rescale_single_result, results[key]))
        return results

    def _rescale_single_result(self, polygon, scale_factor):
        point_num = len(polygon)
        assert point_num % 2 == 0
        polygon = (np.array(polygon) *
                   (np.tile(scale_factor[:2], int(point_num / 2)).reshape(
                       1, -1))).flatten().tolist()
        return polygon

    def merge_text_spotter_result(self, det_result, recog_results):
        results = dict()
        results.update(det_result)
        results.update(recog_results)
        return results

    def _forward_single(self,
                        det_pred_result,
                        recog_pred_result,
                        scale_factor=None,
                        rescale=True,
                        property=None,
                        rescale_extra_property=False,
                        extra_property=None,
                        **kwargs):

        if rescale:
            det_pred_result = self.rescale_results(det_pred_result,
                                                   scale_factor, property)

        if rescale_extra_property and extra_property is not None:
            for key in extra_property:
                assert key in det_pred_result
            det_pred_result = self.rescale_results(det_pred_result,
                                                   scale_factor,
                                                   extra_property)
        results = self.merge_text_spotter_result(det_pred_result,
                                                 recog_pred_result)
        return results

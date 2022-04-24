# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial

import numpy as np
import torch.nn as nn

# from mmocr.utils import check_argument


class BasePostprocessor:

    def __init__(self, text_repr_type='poly', **kwargs):
        assert text_repr_type in ['poly', 'quad'
                                  ], f'Invalid text repr type {text_repr_type}'

        self.text_repr_type = text_repr_type

    def is_valid_instance(self, area, confidence, area_thresh,
                          confidence_thresh):

        return bool(area >= area_thresh and confidence > confidence_thresh)


class BaseTextDetPostProcessor(nn.Module):
    """For FCOS head only."""

    def __init__(self,
                 text_repr_type='poly',
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        super(BaseTextDetPostProcessor, self).__init__()
        assert text_repr_type in ['poly', 'quad']
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def forward(self, pred_results, img_metas=None, **kwargs):

        cfg = self.train_cfg if self.training else self.test_cfg
        assert cfg is not None
        cfg.update(kwargs)
        if len(img_metas) == 1:
            img_metas = [img_metas]
        pred_results = self.split_results(pred_results, img_metas)
        forward_single = partial(self._forward_single, **cfg)
        results = list(map(forward_single, pred_results, img_metas))

        return results

    def _forward_single(self,
                        pred_result,
                        img_meta=None,
                        rescale_fields=[],
                        **kwargs):

        results = self.get_text_instances(pred_result, img_meta, **kwargs)

        if len(rescale_fields) > 0:
            assert isinstance(rescale_fields, list)
            assert set(rescale_fields).issubset(set(results.keys()))
            results = self.rescale(results, img_meta['scale_factor'],
                                   rescale_fields)
        return results

    def rescale(self, results, scale_factor, rescale_fields=None):
        """Rescale results via scale_factor.

        Args:
            scale_factor (tuple(int)): (w_scale, h_scale, w_scale, h_scale)
        """
        assert isinstance(scale_factor, np.ndarray)
        assert scale_factor.shape[0] == 4
        _rescale_single_result = partial(
            self.rescale_polygons, scale_factor=scale_factor)
        for key in rescale_fields:
            results[key] = list(map(_rescale_single_result, results[key]))
        return results

    def rescale_polygons(self, polygons, scale_factor):
        """Rescale polygons via scale_factor.

        Args:
            polygons (list[float] or list[list[float]]): Polygon(s) in the
                shape of (2k,) or (N, 2k). Each polygon is written in
                [x1, y1, x2, y2, ...].
            scale_factor (tuple(int)): (w_scale, h_scale, w_scale, h_scale)
        """
        polygons = np.array(polygons)
        poly_shape = polygons.shape
        reshape_polygon = polygons.reshape(-1, 2)
        polygons = (reshape_polygon /
                    scale_factor[:2][None]).reshape(poly_shape)
        return polygons.tolist()

    def get_text_instances(self, pred_results, **kwargs):
        """Get text instance predictions of one image."""
        raise NotImplementedError

    def split_results(self, pred_results, img_metas, **kwargs):
        """Convert pred_results to the follow format:

        Args:
            pred_results (list[dict]): The list size is batch size. The dict
                contains the prediction result of a single image.
        """
        return pred_results

# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import numpy as np
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.formating import DefaultFormatBundle

import mmocr.utils as utils
from mmocr.core.visualize import (overlay_mask_img, overlay_mask_img_bezier,
                                  show_feature)


@PIPELINES.register_module()
class CustomFormatBundle(DefaultFormatBundle):
    """Custom formatting bundle.

    It formats common fields such as 'img' and 'proposals' as done in
    DefaultFormatBundle, while other fields such as 'gt_kernels' and
    'gt_effective_region_mask' will be formatted to DC as follows:

    - gt_kernels: to DataContainer (cpu_only=True)
    - gt_effective_mask: to DataContainer (cpu_only=True)

    Args:
        keys (list[str]): Fields to be formatted to DC only.
        call_super (bool): If True, format common fields
            by DefaultFormatBundle, else format fields in keys above only.
        visualize (dict): If flag=True, visualize gt mask for debugging.
    """

    def __init__(self,
                 keys=[],
                 call_super=True,
                 visualize=dict(
                     flag=False, boundary_key=None, boundary_type=None)):

        super().__init__()
        self.visualize = visualize
        self.keys = keys
        self.call_super = call_super

    def __call__(self, results):

        if self.visualize['flag']:
            img = results['img'].astype(np.uint8)
            boundary_key = self.visualize['boundary_key']
            boundary_type = self.visualize.get('boundary_type', None)
            features = []
            names = []
            to_uint8 = []
            if boundary_key is not None:
                if not utils.is_type_list(boundary_key, str):
                    boundary_key = [boundary_key]
                if boundary_type is None:
                    boundary_type = ['mask' for _ in range(len(boundary_key))]
                if not utils.is_type_list(boundary_type, str):
                    boundary_type = [boundary_type]
                assert len(boundary_type) == len(boundary_key)
                for i, k in enumerate(boundary_key):
                    if boundary_type[i] == 'bezier':
                        out_img = overlay_mask_img_bezier(
                            deepcopy(img), results[k])
                    else:
                        out_img = overlay_mask_img(deepcopy(img), results[k])
                    features.append(out_img)
                    names.append(k + '_img_' + str(i))
                    to_uint8.append(1)

            show_feature(features, names, to_uint8)

        if self.call_super:
            results = super().__call__(results)

        for k in self.keys:
            results[k] = DC(results[k], cpu_only=True)

        return results

    def __repr__(self):
        return self.__class__.__name__

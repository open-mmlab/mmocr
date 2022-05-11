# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines.formatting import DefaultFormatBundle

from mmocr.core.visualize import overlay_mask_img, show_feature
from mmocr.registry import TRANSFORMS


@TRANSFORMS.register_module()
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
                 visualize=dict(flag=False, boundary_key=None)):

        super().__init__()
        self.visualize = visualize
        self.keys = keys
        self.call_super = call_super

    def __call__(self, results):

        if self.visualize['flag']:
            img = results['img'].astype(np.uint8)
            boundary_key = self.visualize['boundary_key']
            if boundary_key is not None:
                img = overlay_mask_img(img, results[boundary_key].masks[0])

            features = [img]
            names = ['img']
            to_uint8 = [1]

            for k in results['mask_fields']:
                for iter in range(len(results[k].masks)):
                    features.append(results[k].masks[iter])
                    names.append(k + str(iter))
                    to_uint8.append(0)
            show_feature(features, names, to_uint8)

        if self.call_super:
            results = super().__call__(results)

        for k in self.keys:
            results[k] = DC(results[k], cpu_only=True)

        return results

    def __repr__(self):
        return self.__class__.__name__

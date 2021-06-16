import numpy as np
from mmcv import rescale_size
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.formating import DefaultFormatBundle, to_tensor


@PIPELINES.register_module()
class ResizeNoImg:
    """Image resizing without img.

    Used for KIE.
    """

    def __init__(self, img_scale, keep_ratio=True):
        self.img_scale = img_scale
        self.keep_ratio = keep_ratio

    def __call__(self, results):
        w, h = results['img_info']['width'], results['img_info']['height']
        if self.keep_ratio:
            (new_w, new_h) = rescale_size((w, h),
                                          self.img_scale,
                                          return_scale=False)
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            (new_w, new_h) = self.img_scale

        w_scale = new_w / w
        h_scale = new_h / h
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        results['img_shape'] = (new_h, new_w, 1)
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = True

        return results


@PIPELINES.register_module()
class KIEFormatBundle(DefaultFormatBundle):
    """Key information extraction formatting bundle.

    Based on the DefaultFormatBundle, itt simplifies the pipeline of formatting
    common fields, including "img", "proposals", "gt_bboxes", "gt_labels",
    "gt_masks", "gt_semantic_seg", "relations" and "texts".
    These fields are formatted as follows.

    - img: (1) transpose, (2) to tensor, (3) to DataContainer (stack=True)
    - proposals: (1) to tensor, (2) to DataContainer
    - gt_bboxes: (1) to tensor, (2) to DataContainer
    - gt_bboxes_ignore: (1) to tensor, (2) to DataContainer
    - gt_labels: (1) to tensor, (2) to DataContainer
    - gt_masks: (1) to tensor, (2) to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1) unsqueeze dim-0 (2) to tensor,
                       (3) to DataContainer (stack=True)
    - relations: (1) scale, (2) to tensor, (3) to DataContainer
    - texts: (1) to tensor, (2) to DataContainer
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        super().__call__(results)
        if 'ann_info' in results:
            for key in ['relations', 'texts']:
                value = results['ann_info'][key]
                if key == 'relations' and 'scale_factor' in results:
                    scale_factor = results['scale_factor']
                    if isinstance(scale_factor, float):
                        sx = sy = scale_factor
                    else:
                        sx, sy = results['scale_factor'][:2]
                    r = sx / sy
                    factor = np.array([sx, sy, r, 1, r]).astype(np.float32)
                    value = value * factor[None, None]
                results[key] = DC(to_tensor(value))
        return results

    def __repr__(self):
        return self.__class__.__name__

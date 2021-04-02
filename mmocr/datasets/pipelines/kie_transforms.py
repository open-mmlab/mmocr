import numpy as np
from mmcv.parallel import DataContainer as DC

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.formating import DefaultFormatBundle, to_tensor


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
    - gt_semantic_seg: (1) unsqueeze dim-0 (2) to tensor, \
                       (3) to DataContainer (stack=True)
    - relations: (1) scale, (2) to tensor, (3) to DataContainer
    - texts: (1) to tensor, (2) to DataContainer
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with \
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
                    value = value * np.array([sx, sy, r, 1, r])[None, None]
                results[key] = DC(to_tensor(value))
        return results

    def __repr__(self):
        return self.__class__.__name__

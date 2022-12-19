# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Union

import numpy as np


def remove_pipeline_elements(results: Dict,
                             remove_inds: Union[List[int],
                                                np.ndarray]) -> Dict:
    """Remove elements in the pipeline given target indexes.

    Args:
        results (dict): Result dict from loading pipeline.
        remove_inds (list(int) or np.ndarray): The element indexes to be
            removed.

    Required Keys:

    - gt_polygons (optional)
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignored (optional)
    - gt_texts (optional)

    Modified Keys:

    - gt_polygons (optional)
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignored (optional)
    - gt_texts (optional)

    Returns:
        dict: The results with element removed.
    """
    keys = [
        'gt_polygons', 'gt_bboxes', 'gt_bboxes_labels', 'gt_ignored',
        'gt_texts'
    ]
    num_elements = -1
    for key in keys:
        if key in results:
            num_elements = len(results[key])
            break
    if num_elements == -1:
        return results
    kept_inds = np.array(
        [i for i in range(num_elements) if i not in remove_inds])
    for key in keys:
        if key in results:
            if results[key] is np.ndarray:
                results[key] = results[key][kept_inds]
            elif results[key] is list:
                results[key] = [results[key][i] for i in kept_inds]
            else:
                raise NotImplementedError
    return results

# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations

from mmocr.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadOCRAnnotations(MMCV_LoadAnnotations):
    """Load and process the ``instances`` annotation provided by dataset.

    The annotation format is as the following:

    .. code-block:: python

        {
            'instances':
            [
                {
                # List of 4 numbers representing the bounding box of the
                # instance, in (x1, y1, x2, y2) order.
                # used in text detection or text spotting tasks.
                'bbox': [x1, y1, x2, y2],

                # Label of instance, usually it's 0.
                # used in text detection or text spotting tasks.
                'bbox_label': 0,

                # List of n numbers representing the polygon of the
                # instance, in (xn, yn) order.
                # used in text detection/ textspotter.
                "polygon": [x1, y1, x2, y2, ... xn, yn],

                # The flag indicating whether the instance should be ignored.
                # used in text detection or text spotting tasks.
                "ignore": False,

                # The groundtruth of text.
                # used in text recognition or text spotting tasks.
                "text": 'tmp',
                }
            ]
        }

    After this module, the annotation has been changed to the format below:

    .. code-block:: python

        {
            # In (x1, y1, x2, y2) order, float type. N is the number of bboxes
            # in np.float32
            'gt_bboxes': np.ndarray(N, 4)
             # In np.int64 type.
            'gt_bboxes_labels': np.ndarray(N, )
            # In (x1, y1,..., xk, yk) order, float type.
            # in list[np.float32]
            'gt_polygons': list[np.ndarray(2k, )]
             # In np.bool_ type.
            'gt_ignored': np.ndarray(N, )
             # In list[str]
            'gt_texts': list[str]
        }

    Required Keys:

    - instances

      - bbox (optional)
      - bbox_label (optional)
      - polygon (optional)
      - ignore (optional)
      - text (optional)

    Added Keys:

    - gt_bboxes (np.float32)
    - gt_bboxes_labels (np.int64)
    - gt_polygons (list[np.float32])
    - gt_ignored (np.bool_)
    - gt_text (list[str])

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
            Defaults to False.
        with_label (bool): Whether to parse and load the label annotation.
            Defaults to False.
        with_polygon (bool): Whether to parse and load the polygon annotation.
            Defaults to False.
        with_text (bool): Whether to parse and load the text annotation.
            Defaults to False.
    """

    def __init__(self,
                 with_bbox: bool = False,
                 with_label: bool = False,
                 with_polygon: bool = False,
                 with_text: bool = False,
                 **kwargs) -> None:
        super().__init__(with_bbox=with_bbox, with_label=with_label, **kwargs)
        self.with_polygon = with_polygon
        self.with_text = with_text
        self.with_ignore = with_bbox or with_polygon

    def _load_ignore_flags(self, results: dict) -> None:
        """Private function to load ignore annotations.

        Args:
            results (dict): Result dict from :obj:``OCRDataset``.

        Returns:
            dict: The dict contains loaded ignore annotations.
        """
        gt_ignored = []
        for instance in results['instances']:
            gt_ignored.append(instance['ignore'])
        results['gt_ignored'] = np.array(gt_ignored, dtype=np.bool_)

    def _load_polygons(self, results: dict) -> None:
        """Private function to load polygon annotations.

        Args:
            results (dict): Result dict from :obj:``OCRDataset``.

        Returns:
            dict: The dict contains loaded polygon annotations.
        """

        gt_polygons = []
        for instance in results['instances']:
            gt_polygons.append(np.array(instance['polygon'], dtype=np.float32))
        results['gt_polygons'] = gt_polygons

    def _load_texts(self, results: dict) -> None:
        """Private function to load text annotations.

        Args:
            results (dict): Result dict from :obj:``OCRDataset``.

        Returns:
            dict: The dict contains loaded text annotations.
        """
        gt_texts = []
        for instance in results['instances']:
            gt_texts.append(instance['text'])
        results['gt_texts'] = gt_texts

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:``OCRDataset``.

        Returns:
            dict: The dict contains loaded bounding box, label polygon and
            text annotations.
        """
        results = super().transform(results)
        if self.with_polygon:
            self._load_polygons(results)
        if self.with_text:
            self._load_texts(results)
        if self.with_ignore:
            self._load_ignore_flags(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_polygon={self.with_polygon}, '
        repr_str += f'with_text={self.with_text}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'file_client_args={self.file_client_args})'
        return repr_str

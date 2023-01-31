import random
from typing import Dict

import numpy as np
from mmcv.transforms.base import BaseTransform

from mmocr.datasets.transforms import LoadOCRAnnotations
from mmocr.registry import TASK_UTILS, TRANSFORMS
from mmocr.utils import bezier2poly, poly2bezier


@TRANSFORMS.register_module()
class LoadOCRAnnotationsWithBezier(LoadOCRAnnotations):
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
    - gt_texts (list[str])

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
            Defaults to False.
        with_label (bool): Whether to parse and load the label annotation.
            Defaults to False.
        with_polygon (bool): Whether to parse and load the polygon annotation.
            Defaults to False.
        with_bezier (bool): Whether to parse and load the bezier annotation.
            Defaults to False.
        with_text (bool): Whether to parse and load the text annotation.
            Defaults to False.
    """

    def __init__(self, with_bezier: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.with_bezier = with_bezier

    def _load_beziers(self, results: dict) -> None:
        """Private function to load text annotations.

        Args:
            results (dict): Result dict from :obj:``OCRDataset``.

        Returns:
            dict: The dict contains loaded text annotations.
        """
        gt_beziers = []
        for instance in results['instances']:
            gt_beziers.append(instance['beziers'])
        results['gt_beziers'] = gt_beziers

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:``OCRDataset``.

        Returns:
            dict: The dict contains loaded bounding box, label polygon and
            text annotations.
        """
        results = super().transform(results)
        if self.with_bezier:
            self._load_beziers(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_polygon={self.with_polygon}, '
        repr_str += f'with_bezier={self.with_bezier}, '
        repr_str += f'with_text={self.with_text}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'file_client_args={self.file_client_args})'
        return repr_str


@TRANSFORMS.register_module()
class Bezier2Polygon(BaseTransform):
    """Convert bezier curves to polygon.

    Required Keys:

    - gt_beziers

    Modified Keys:

    - gt_polygons
    """

    def transform(self, results: Dict) -> Dict:
        """

        Args:
            results (dict): Result dict containing the data to transform.

        Returns:
            Optional[dict]: The transformed data. If all the polygons are
            unfixable, return None.
        """
        if results.get('gt_beziers', None) is not None:
            results['gt_polygons'] = [
                np.array(bezier2poly(poly), dtype=np.float32)
                for poly in results['gt_beziers']
            ]
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += '()'
        return repr_str


@TRANSFORMS.register_module()
class Polygon2Bezier(BaseTransform):
    """Convert polygons to bezier curves.

    Required Keys:

    - gt_polygons

    Added Keys:

    - gt_beziers
    """

    def transform(self, results: Dict) -> Dict:
        """

        Args:
            results (dict): Result dict containing the data to transform.

        Returns:
            Optional[dict]: The transformed data. If all the polygons are
            unfixable, return None.
        """
        if results.get('gt_polygons', None) is not None:
            beziers = [poly2bezier(poly) for poly in results['gt_polygons']]
            results['gt_beziers'] = np.array(beziers, dtype=np.float32)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += '()'
        return repr_str


@TRANSFORMS.register_module()
class ConvertText(BaseTransform):

    def __init__(self, dictionary):
        if isinstance(dictionary, dict):
            self.dictionary = TASK_UTILS.build(dictionary)
        else:
            raise TypeError(
                'The type of dictionary should be `Dictionary` or dict, '
                f'but got {type(dictionary)}')

    def transform(self, results: Dict) -> Dict:
        """

        Args:
            results (dict): Result dict containing the data to transform.

        Returns:
            Optional[dict]: The transformed data. If all the polygons are
            unfixable, return None.
        """
        new_gt_texts = []
        for gt_text in results['gt_texts']:
            if self.dictionary.end_idx in gt_text:
                gt_text = gt_text[:gt_text.index(self.dictionary.end_idx)]
            new_gt_texts.append(self.dictionary.idx2str(gt_text))
        results['gt_texts'] = new_gt_texts
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += '()'
        return repr_str


@TRANSFORMS.register_module()
class RescaleToShortSide(BaseTransform):

    def __init__(self,
                 short_side_lens,
                 long_side_bound,
                 resize_type: str = 'Resize',
                 **resize_kwargs):
        self.short_side_lens = short_side_lens
        self.long_side_bound = long_side_bound
        self.resize_cfg = dict(type=resize_type, **resize_kwargs)

        # create a empty Reisize object
        self.resize_cfg.update(dict(scale=0))
        self.resize = TRANSFORMS.build(self.resize_cfg)

    def transform(self, results: Dict) -> Dict:
        """Resize image.

        Args:
            results (dict): Result dict containing the data to transform.

        Returns:
            Optional[dict]: The transformed data. If all the polygons are
            unfixable, return None.
        """
        short_len = random.choice(self.short_side_lens)
        new_h, new_w = self.get_size_with_aspect_ratio(results['img_shape'],
                                                       short_len,
                                                       self.long_side_bound)
        self.resize.scale = (new_w, new_h)
        return self.resize(results)

    def get_size_with_aspect_ratio(self, image_size, size, max_size=None):
        h, w = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(
                    round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(short_side_lens={self.short_side_lens}, '
        repr_str += f'long_side_bound={self.long_side_bound}, '
        repr_str += f'resize_cfg={self.resize_cfg})'
        return repr_str

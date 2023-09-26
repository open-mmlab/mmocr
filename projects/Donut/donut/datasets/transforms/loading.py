import copy
from typing import Optional, Union
import numpy as np
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations

from mmocr.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadJsonAnnotations(MMCV_LoadAnnotations):
    """Load and process the ``instances`` annotation provided by dataset.

    The annotation format is as the following:

    .. code-block:: python

        {
            # A nested list of 4 numbers representing the bounding box of the
            # instance, in (x1, y1, x2, y2) order.
            'bbox': np.array([[x1, y1, x2, y2], [x1, y1, x2, y2], ...],
                             dtype=np.int32),

            # Labels of boxes. Shape is (N,).
            'bbox_labels': np.array([0, 2, ...], dtype=np.int32),

            # Labels of edges. Shape (N, N).
            'edge_labels': np.array([0, 2, ...], dtype=np.int32),

            # List of texts.
            "texts": ['text1', 'text2', ...],
        }

    After this module, the annotation has been changed to the format below:

    .. code-block:: python

        {
            # In (x1, y1, x2, y2) order, float type. N is the number of bboxes
            # in np.float32
            'gt_bboxes': np.ndarray(N, 4),
            # In np.int64 type.
            'gt_bboxes_labels': np.ndarray(N, ),
            # In np.int32 type.
            'gt_edges_labels': np.ndarray(N, N),
            # In list[str]
            'gt_texts': list[str],
            # tuple(int)
            'ori_shape': (H, W)
        }

    Required Keys:

    - bboxes
    - bbox_labels
    - edge_labels
    - texts

    Added Keys:

    - gt_bboxes (np.float32)
    - gt_bboxes_labels (np.int64)
    - gt_edges_labels (np.int64)
    - gt_texts (list[str])
    - ori_shape (tuple[int])

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
            Defaults to True.
        with_label (bool): Whether to parse and load the label annotation.
            Defaults to True.
        with_text (bool): Whether to parse and load the text annotation.
            Defaults to True.
        directed (bool): Whether build edges as a directed graph.
            Defaults to False.
        key_node_idx (int, optional): Key node label, used to mask out edges
            that are not connected from key nodes to value nodes. It has to be
            specified together with ``value_node_idx``. Defaults to None.
        value_node_idx (int, optional): Value node label, used to mask out
            edges that are not connected from key nodes to value nodes. It has
            to be specified together with ``key_node_idx``. Defaults to None.
    """

    def __init__(self,
                 with_bbox: bool = True,
                 with_label: bool = True,
                 with_text: bool = True,
                 directed: bool = False,
                 key_node_idx: Optional[int] = None,
                 value_node_idx: Optional[int] = None,
                 **kwargs) -> None:
        super().__init__(with_bbox=with_bbox, with_label=with_label, **kwargs)
        self.with_text = with_text
        self.directed = directed
        if key_node_idx is not None or value_node_idx is not None:
            assert key_node_idx is not None and value_node_idx is not None
            self.key_node_idx = key_node_idx
            self.value_node_idx = value_node_idx

    def _load_parse(self, results: dict) -> None:
        """Private function to load text annotations.

        Args:
            results (dict): Result dict from :obj:``OCRDataset``.
        """
        gt_parse = []
        for instance in results['instances']:
            gt_parse.append(instance['parses_json'])
        results['parses_json'] = gt_parse

    def _load_labels(self, results: dict) -> None:
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:``WildReceiptDataset``.
        """
        bbox_labels = []
        edge_labels = []
        for instance in results['instances']:
            bbox_labels.append(instance['bbox_label'])
            edge_labels.append(instance['edge_label'])

        bbox_labels = np.array(bbox_labels, np.int32)
        edge_labels = np.array(edge_labels)
        edge_labels = (edge_labels[:, None] == edge_labels[None, :]).astype(
            np.int32)

        if self.directed:
            edge_labels = (edge_labels & bbox_labels == 1).astype(np.int32)

        if hasattr(self, 'key_node_idx'):
            key_nodes_mask = bbox_labels == self.key_node_idx
            value_nodes_mask = bbox_labels == self.value_node_idx
            key2value_mask = key_nodes_mask[:,
                                            None] * value_nodes_mask[None, :]
            edge_labels[~key2value_mask] = -1

        np.fill_diagonal(edge_labels, -1)

        results['gt_edges_labels'] = edge_labels.astype(np.int64)
        results['gt_bboxes_labels'] = bbox_labels.astype(np.int64)

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:``OCRDataset``.

        Returns:
            dict: The dict contains loaded bounding box, label polygon and
            text annotations.
        """
        if 'ori_shape' not in results:
            results['ori_shape'] = copy.deepcopy(results['img_shape'])
        results = super().transform(results)
        self._load_parse(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label})'
        return repr_str

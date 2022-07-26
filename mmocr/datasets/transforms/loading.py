# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
from typing import Optional

import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations
from mmcv.transforms import LoadImageFromFile as MMCV_LoadImageFromFile

from mmocr.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadImageFromFile(MMCV_LoadImageFromFile):
    """Load an image from file.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:``mmcv.imfrombytes``.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :func:``mmcv.imfrombytes`` for details.
            Defaults to 'cv2'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        min_size (int): The minimum size of the image to be loaded. If the
            image is smaller than the minimum size, it will be ignored.
            Defaults to 0.
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: dict = dict(backend='disk'),
                 min_size: int = 0,
                 ignore_empty: bool = False) -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        self.file_client_args = file_client_args.copy()
        self.file_client = mmcv.FileClient(**self.file_client_args)
        self.min_size = min_size

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.
        """
        results = super().transform(results)
        if results and min(results['ori_shape']) < self.min_size:
            return None
        else:
            return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'min_size={self.min_size}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


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
    - gt_texts (list[str])

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


@TRANSFORMS.register_module()
class LoadKIEAnnotations(MMCV_LoadAnnotations):
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

    def _load_texts(self, results: dict) -> None:
        """Private function to load text annotations.

        Args:
            results (dict): Result dict from :obj:``OCRDataset``.
        """
        gt_texts = []
        for instance in results['instances']:
            gt_texts.append(instance['text'])
        results['gt_texts'] = gt_texts

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
        if self.with_text:
            self._load_texts(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_text={self.with_text})'
        return repr_str


@TRANSFORMS.register_module()
class LoadImageFromLMDB(BaseTransform):
    """Load an image from lmdb file. Only support LMDB file at disk.

    LMDB file is organized with the following structure:
        lmdb
            |__data.mdb
            |__lock.mdb

    Required Keys:

    - img_path (In LMDB img_path is a key in the format of "image-{i:09d}".)

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:``mmcv.imfrombytes``.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :func:``mmcv.imfrombytes`` for details.
            Defaults to 'cv2'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='lmdb', db_path='')``.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: dict = dict(backend='lmdb', db_path=''),
                 ignore_empty: bool = False) -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        self.file_client_args = file_client_args.copy()
        self.file_client = mmcv.FileClient(**self.file_client_args)

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image from LMDB file.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        filename = results['img_path']
        lmdb_path = os.path.dirname(filename)
        image_key = os.path.basename(filename)
        self.file_client.client.db_path = lmdb_path
        img_bytes = self.file_client.get(image_key)
        if img_bytes is None:
            return None
        try:
            img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        except OSError:
            return None
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str

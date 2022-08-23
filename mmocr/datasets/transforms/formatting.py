# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import InstanceData, LabelData

from mmocr.registry import TRANSFORMS
from mmocr.structures import (KIEDataSample, TextDetDataSample,
                              TextRecogDataSample)


@TRANSFORMS.register_module()
class PackTextDetInputs(BaseTransform):
    """Pack the inputs data for text detection.

    The type of outputs is `dict`:

    - inputs: image converted to tensor, whose shape is (C, H, W).
    - data_sample: Two components of ``TextDetDataSample`` will be updated:

      - gt_instances (InstanceData): Depending on annotations, a subset of the
        following keys will be updated:

        - bboxes (torch.Tensor((N, 4), dtype=torch.float32)): The groundtruth
          of bounding boxes in the form of [x1, y1, x2, y2]. Renamed from
          'gt_bboxes'.
        - labels (torch.LongTensor(N)): The labels of instances.
          Renamed from 'gt_bboxes_labels'.
        - polygons(list[np.array((2k,), dtype=np.float32)]): The
          groundtruth of polygons in the form of [x1, y1,..., xk, yk]. Each
          element in polygons may have different number of points. Renamed from
          'gt_polygons'. Using numpy instead of tensor is that polygon usually
          is not the output of model and operated on cpu.
        - ignored (torch.BoolTensor((N,))): The flag indicating whether the
          corresponding instance should be ignored. Renamed from
          'gt_ignored'.
        - texts (list[str]): The groundtruth texts. Renamed from 'gt_texts'.

      - metainfo (dict): 'metainfo' is always populated. The contents of the
        'metainfo' depends on ``meta_keys``. By default it includes:

        - "img_path": Path to the image file.
        - "img_shape": Shape of the image input to the network as a tuple
          (h, w). Note that the image may be zero-padded afterward on the
          bottom/right if the batch tensor is larger than this shape.
        - "scale_factor": A tuple indicating the ratio of width and height
          of the preprocessed image to the original one.
        - "ori_shape": Shape of the preprocessed image as a tuple
          (h, w).
        - "pad_shape": Image shape after padding (if any Pad-related
          transform involved) as a tuple (h, w).
        - "flip": A boolean indicating if the image has been flipped.
        - ``flip_direction``: the flipping direction.

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            the metainfo of ``TextDetSample``. Defaults to ``('img_path',
            'ori_shape', 'img_shape', 'scale_factor', 'flip',
            'flip_direction')``.
    """
    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_polygons': 'polygons',
        'gt_texts': 'texts',
        'gt_ignored': 'ignored'
    }

    def __init__(self,
                 meta_keys=('img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'flip', 'flip_direction')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): Data for model forwarding.
            - 'data_sample' (obj:`DetDataSample`): The annotation info of the
              sample.
        """
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            packed_results['inputs'] = to_tensor(img)

        data_sample = TextDetDataSample()
        instance_data = InstanceData()
        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key in ['gt_bboxes', 'gt_bboxes_labels', 'gt_ignored']:
                instance_data[self.mapping_table[key]] = to_tensor(
                    results[key])
            else:
                instance_data[self.mapping_table[key]] = results[key]
        data_sample.gt_instances = instance_data

        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_sample'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str


@TRANSFORMS.register_module()
class PackTextRecogInputs(BaseTransform):
    """Pack the inputs data for text recognition.

    The type of outputs is `dict`:

    - inputs: Image as a tensor, whose shape is (C, H, W).
    - data_sample: Two components of ``TextRecogDataSample`` will be updated:

      - gt_text (LabelData):

        - item(str): The groundtruth of text. Rename from 'gt_texts'.

      - metainfo (dict): 'metainfo' is always populated. The contents of the
        'metainfo' depends on ``meta_keys``. By default it includes:

        - "img_path": Path to the image file.
        - "ori_shape":  Shape of the preprocessed image as a tuple
          (h, w).
        - "img_shape": Shape of the image input to the network as a tuple
          (h, w). Note that the image may be zero-padded afterward on the
          bottom/right if the batch tensor is larger than this shape.
        - "valid_ratio": The proportion of valid (unpadded) content of image
          on the x-axis. It defaults to 1 if not set in pipeline.

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            the metainfo of ``TextRecogDataSampel``. Defaults to
            ``('img_path', 'ori_shape', 'img_shape', 'pad_shape',
            'valid_ratio')``.
    """

    def __init__(self,
                 meta_keys=('img_path', 'ori_shape', 'img_shape', 'pad_shape',
                            'valid_ratio')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): Data for model forwarding.
            - 'data_sample' (obj:`TextRecogDataSample`): The annotation info
                of the sample.
        """
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            packed_results['inputs'] = to_tensor(img)

        data_sample = TextRecogDataSample()
        gt_text = LabelData()

        if results.get('gt_texts', None):
            assert len(
                results['gt_texts']
            ) == 1, 'Each image sample should have one text annotation only'
            gt_text.item = results['gt_texts'][0]
        data_sample.gt_text = gt_text

        img_meta = {}
        for key in self.meta_keys:
            if key == 'valid_ratio':
                img_meta[key] = results.get('valid_ratio', 1)
            else:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)

        packed_results['data_sample'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str


@TRANSFORMS.register_module()
class PackKIEInputs(BaseTransform):
    """Pack the inputs data for key information extraction.

    The type of outputs is `dict`:

    - inputs: image converted to tensor, whose shape is (C, H, W).
    - data_sample: Two components of ``TextDetDataSample`` will be updated:

      - gt_instances (InstanceData): Depending on annotations, a subset of the
        following keys will be updated:

        - bboxes (torch.Tensor((N, 4), dtype=torch.float32)): The groundtruth
          of bounding boxes in the form of [x1, y1, x2, y2]. Renamed from
          'gt_bboxes'.
        - labels (torch.LongTensor(N)): The labels of instances.
          Renamed from 'gt_bboxes_labels'.
        - edge_labels (torch.LongTensor(N, N)): The edge labels.
          Renamed from 'gt_edges_labels'.
        - texts (list[str]): The groundtruth texts. Renamed from 'gt_texts'.

      - metainfo (dict): 'metainfo' is always populated. The contents of the
        'metainfo' depends on ``meta_keys``. By default it includes:

        - "img_path": Path to the image file.
        - "img_shape": Shape of the image input to the network as a tuple
          (h, w). Note that the image may be zero-padded afterward on the
          bottom/right if the batch tensor is larger than this shape.
        - "scale_factor": A tuple indicating the ratio of width and height
          of the preprocessed image to the original one.
        - "ori_shape": Shape of the preprocessed image as a tuple
          (h, w).

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            the metainfo of ``TextDetSample``. Defaults to ``('img_path',
            'ori_shape', 'img_shape', 'scale_factor', 'flip',
            'flip_direction')``.
    """
    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_edges_labels': 'edge_labels',
        'gt_texts': 'texts',
    }

    def __init__(self,
                 meta_keys=('img_path', 'ori_shape', 'img_shape',
                            'scale_factor')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): Data for model forwarding.
            - 'data_sample' (obj:`DetDataSample`): The annotation info of the
              sample.
        """
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            packed_results['inputs'] = to_tensor(img)
        else:
            packed_results['inputs'] = torch.FloatTensor().reshape(0, 0, 0)

        data_sample = KIEDataSample()
        instance_data = InstanceData()
        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key in ['gt_bboxes', 'gt_bboxes_labels', 'gt_edges_labels']:
                instance_data[self.mapping_table[key]] = to_tensor(
                    results[key])
            else:
                instance_data[self.mapping_table[key]] = results[key]
        data_sample.gt_instances = instance_data

        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_sample'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str

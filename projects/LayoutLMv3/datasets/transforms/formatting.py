# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import LabelData

from mmocr.registry import TRANSFORMS
from projects.LayoutLMv3.structures import SERDataSample


@TRANSFORMS.register_module()
class PackSERInputs(BaseTransform):
    """Pack the inputs data for LayoutLMv3ForTokenClassification model.

    The type of outputs is `dict`:

    - inputs: Data for model forwarding. Five components will be included:

      - input_ids, whose shape is (truncation_number, 512).
      - bbox, whose shape is (truncation_number, 512, 4).
      - attention_mask, whose shape is (truncation_number, 512).
      - pixel_values, whose shape is (truncation_number, 3, 224, 224).
      - labels, whose shape is (truncation_number, 512).

    - data_samples: Two components of ``SERDataSample`` will be updated:

      - gt_instances (InstanceData): Depending on annotations, a subset of the
        following keys will be updated:

        - bboxes (torch.Tensor((N, 4), dtype=torch.float32)): The groundtruth
          of bounding boxes in the form of [x1, y1, x2, y2]. Renamed from
          'gt_bboxes'.
        - labels (torch.LongTensor(N)): The labels of instances.
          Renamed from 'gt_bboxes_labels'.
        - texts (list[str]): The groundtruth texts. Renamed from 'gt_texts'.

      - metainfo (dict): 'metainfo' is always populated. The contents of the
        'metainfo' depends on ``meta_keys``. By default it includes:

        - "img_path": Path to the image file.
        - "img_shape": Shape of the image input to the network as a tuple
          (h, w). Note that the image may be zero-padded afterward on the
          bottom/right if the batch tensor is larger than this shape.
        - "scale_factor": A tuple indicating the ratio of width and height
          of the preprocessed image to the original one.
        - "ori_shape": Shape of the preprocessed image as a tuple (h, w).

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            the metainfo of ``SERDataSample``. Defaults to ``('img_path',
            'ori_shape', 'img_shape', 'scale_factor')``.
    """
    # HF LayoutLMv3ForTokenClassification model input params.
    ser_keys = [
        'input_ids', 'bbox', 'attention_mask', 'pixel_values', 'labels'
    ]

    def __init__(self, meta_keys=()):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack SER input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`dict`): Data for model forwarding.
            - 'data_samples' (obj:`SERDataSample`): The annotation info of the
              sample.
        """

        packed_results = dict()
        truncation_number = results['truncation_number']

        if 'pixel_values' in results:
            img = results['pixel_values']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            # A simple trick to speedup formatting by 3-5 times when
            # OMP_NUM_THREADS != 1
            # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
            # for more details
            if img.flags.c_contiguous:
                img = to_tensor(img)
                img = img.permute(2, 0, 1).contiguous()
            else:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                img = to_tensor(img)
            results['pixel_values'] = torch.cat(
                [img.unsqueeze(0)] * truncation_number, dim=0)

        # pack `inputs`
        inputs = {}
        for key in self.ser_keys:
            if key not in results:
                continue
            inputs[key] = to_tensor(results[key])
        packed_results['inputs'] = inputs

        # pack `data_samples`
        data_samples = []
        for truncation_idx in range(truncation_number):
            data_sample = SERDataSample()
            gt_label = LabelData()
            if results.get('labels', None):
                gt_label.item = to_tensor(results['labels'][truncation_idx])
            data_sample.gt_label = gt_label
            meta = {}
            for key in self.meta_keys:
                if key == 'truncation_word_ids':
                    meta[key] = results[key][truncation_idx]
                else:
                    meta[key] = results[key]
            data_sample.set_metainfo(meta)
            data_samples.append(data_sample)
        packed_results['data_samples'] = data_samples

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str

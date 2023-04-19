# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.structures import BaseDataElement, LabelData


class SERDataSample(BaseDataElement):
    """A data structure interface of MMOCR for Semantic Entity Recognition.
    They are used as interfaces between different components.

    The attributes in ``SERDataSample`` are divided into two parts:

        - ``gt_label``(LabelData): Ground truth label.
        - ``pred_label``(LabelData): predictions label.

    Examples:
         >>> import torch
         >>> import numpy as np
         >>> from mmengine.structures import LabelData
         >>> from mmocr.data import SERDataSample
         >>> # gt_label
         >>> data_sample = SERDataSample()
         >>> img_meta = dict(img_shape=(800, 1196, 3),
         ...                 pad_shape=(800, 1216, 3))
         >>> gt_label = LabelData(metainfo=img_meta)
         >>> gt_label.item = 'mmocr'
         >>> data_sample.gt_label = gt_label
         >>> assert 'img_shape' in data_sample.gt_label.metainfo_keys()
         >>> print(data_sample)
         <SERDataSample(
             META INFORMATION
             DATA FIELDS
             gt_label: <LabelData(
                     META INFORMATION
                     pad_shape: (800, 1216, 3)
                     img_shape: (800, 1196, 3)
                     DATA FIELDS
                     item: 'mmocr'
                 ) at 0x7f21fb1b9190>
         ) at 0x7f21fb1b9880>
         >>> # pred_label
         >>> pred_label = LabelData(metainfo=img_meta)
         >>> pred_label.item = 'mmocr'
         >>> data_sample = SERDataSample(pred_label=pred_label)
         >>> assert 'pred_label' in data_sample
         >>> data_sample = SERDataSample()
         >>> gt_label_data = dict(item='mmocr')
         >>> gt_label = LabelData(**gt_label_data)
         >>> data_sample.gt_label = gt_label
         >>> assert 'gt_label' in data_sample
         >>> assert 'item' in data_sample.gt_label
    """

    @property
    def gt_label(self) -> LabelData:
        """LabelData: ground truth label.
        """
        return self._gt_label

    @gt_label.setter
    def gt_label(self, value: LabelData) -> None:
        """gt_label setter."""
        self.set_field(value, '_gt_label', dtype=LabelData)

    @gt_label.deleter
    def gt_label(self) -> None:
        """gt_label deleter."""
        del self._gt_label

    @property
    def pred_label(self) -> LabelData:
        """LabelData: prediction label.
        """
        return self._pred_label

    @pred_label.setter
    def pred_label(self, value: LabelData) -> None:
        """pred_label setter."""
        self.set_field(value, '_pred_label', dtype=LabelData)

    @pred_label.deleter
    def pred_label(self) -> None:
        """pred_label deleter."""
        del self._pred_label

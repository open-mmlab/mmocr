# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.data import BaseDataElement, LabelData


class TextRecogDataSample(BaseDataElement):
    """A data structure interface of MMOCR for text recognition. They are used
    as interfaces between different components.

    The attributes in ``TextRecogDataSample`` are divided into two parts:

        - ``gt_text``(LabelData): Ground truth text.
        - ``pred_text``(LabelData): predictions text.

    Examples:
         >>> import torch
         >>> import numpy as np
         >>> from mmengine.data import LabelData
         >>> from mmocr.data import TextRecogDataSample
         >>> # gt_text
         >>> data_sample = TextRecogDataSample()
         >>> img_meta = dict(img_shape=(800, 1196, 3),
         ...                 pad_shape=(800, 1216, 3))
         >>> gt_text = LabelData(metainfo=img_meta)
         >>> gt_text.item = 'mmocr'
         >>> data_sample.gt_text = gt_text
         >>> assert 'img_shape' in data_sample.gt_text.metainfo_keys()
         >>> print(data_sample)
        <TextRecogDataSample(
            META INFORMATION
            DATA FIELDS
            gt_text: <LabelData(
                    META INFORMATION
                    pad_shape: (800, 1216, 3)
                    img_shape: (800, 1196, 3)
                    DATA FIELDS
                    item: 'mmocr'
                ) at 0x7f21fb1b9190>
        ) at 0x7f21fb1b9880>
         >>> # pred_text
         >>> pred_text = LabelData(metainfo=img_meta)
         >>> pred_text.item = 'mmocr'
         >>> data_sample = TextRecogDataSample(pred_text=pred_text)
         >>> assert 'pred_text' in data_sample
         >>> data_sample = TextRecogDataSample()
         >>> gt_text_data = dict(item='mmocr')
         >>> gt_text = LabelData(**gt_text_data)
         >>> data_sample.gt_text = gt_text
         >>> assert 'gt_text' in data_sample
         >>> assert 'item' in data_sample.gt_text
    """

    @property
    def gt_text(self) -> LabelData:
        """LabelData: ground truth text.
        """
        return self._gt_text

    @gt_text.setter
    def gt_text(self, value: LabelData) -> None:
        """gt_text setter."""
        self.set_field(value, '_gt_text', dtype=LabelData)

    @gt_text.deleter
    def gt_text(self) -> None:
        """gt_text deleter."""
        del self._gt_text

    @property
    def pred_text(self) -> LabelData:
        """LabelData: prediction text.
        """
        return self._pred_text

    @pred_text.setter
    def pred_text(self, value: LabelData) -> None:
        """pred_text setter."""
        self.set_field(value, '_pred_text', dtype=LabelData)

    @pred_text.deleter
    def pred_text(self) -> None:
        """pred_text deleter."""
        del self._pred_text

from typing import Dict, List, Optional, Sequence, Union
from numbers import Number
import torch.nn as nn
from mmengine.model import ImgDataPreprocessor
from mmocr.registry import MODELS


@MODELS.register_module()
class DonutDataPreprocessor(ImgDataPreprocessor):
    def __init__(self,
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 batch_augments: Optional[List[Dict]] = None) -> None:
        super().__init__(
            mean=mean,
            std=std,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr)
        if batch_augments is not None:
            self.batch_augments = nn.ModuleList(
                [MODELS.build(aug) for aug in batch_augments])
        else:
            self.batch_augments = None

    def forward(self, data: Dict, training: bool = False) -> Dict:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        data = super().forward(data=data, training=training)
        inputs, data_samples = data['inputs'], data['data_samples']

        if data_samples is not None:
            batch_input_shape = tuple(inputs[0].size()[-2:])
            for data_sample in data_samples:

                # valid_ratio = data_sample.valid_ratio * \
                #     data_sample.img_shape[1] / batch_input_shape[1]
                data_sample.set_metainfo(
                    dict(
                        # valid_ratio=valid_ratio,
                        batch_input_shape=batch_input_shape))

        if training and self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(inputs, data_samples)

        # load decoder_input_ids, decoder_labels
        return data

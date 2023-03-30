# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.models.textdet.data_preprocessors import TextDetDataPreprocessor
from mmocr.registry import MODELS


@MODELS.register_module()
class LayoutLMv3DataPreprocessor(TextDetDataPreprocessor):
    """Image pre-processor for LayoutLMv3.

    If you want to get the same processing result as
    LayoutLMv3ImageProcessor in HuggingFace, you need to set
    mean/std to [127.5, 127.5, 127.5], bgr_to_rgb = True,
    and set pipeline Resize backend to `pillow`.

    Like:

    train_pipeline = [
        dict(type='LoadImageFromFile', color_type='color'),
        dict(type='Resize',
             scale=(224, 224),
             backend='pillow'),  # backend=pillow 数值与huggingface对齐
        ...
    ]
    model_cfg = dict(
        ...
        data_preprocessor=dict(
            type='LayoutLMv3DataPreprocessor',
            mean=[127.5, 127.5, 127.5],
            std=[127.5, 127.5, 127.5],
            bgr_to_rgb=True),
        ...
        )

    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the maximum size of current batch with defined
      ``pad_value``. The padding size can be divisible by a defined
      ``pad_size_divisor``
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Do batch augmentations during training.

    Args:
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (Number): The padded pixel value. Defaults to 0.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to RGB.
            Defaults to False.
        batch_augments (list[dict], optional): Batch-level augmentations
    """

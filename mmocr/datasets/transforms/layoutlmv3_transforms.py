# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Union

from mmcv.transforms.base import BaseTransform

from mmocr.registry import TRANSFORMS
from transformers import AutoImageProcessor, LayoutLMv3ImageProcessor
from transformers.file_utils import PaddingStrategy
from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import ChannelDimension
from transformers.tokenization_utils_base import BatchEncoding


@TRANSFORMS.register_module()
class ProcessImageForLayoutLMv3(BaseTransform):
    """A transform to process image for LayoutLMv3, which will use HuggingFace
    `AutoImageProcessor`

    Required Keys:

    - img
    - img_shape

    Modified Keys:

    - img_shape

    Added Keys:

    - scale_factor
    - pixel_values

    Args:
        image_processor (dict): The image_processor cfg, which the key
            `pretrained_model_name_or_path` must be specified.
    """

    image_processor_class = (LayoutLMv3ImageProcessor)

    def __init__(self,
                 image_processor: dict = dict(
                     pretrained_model_name_or_path=None),
                 label_pad_token_id: int = -100) -> None:
        super().__init__()
        if isinstance(image_processor, dict) and \
                image_processor.get('pretrained_model_name_or_path', None):
            self.image_processor = AutoImageProcessor.from_pretrained(
                **image_processor)
        else:
            raise TypeError(
                'image_processor cfg should be a `dict` and a key '
                '`pretrained_model_name_or_path` must be specified')

        if not isinstance(self.image_processor, self.image_processor_class):
            raise ValueError(
                f'Received a {type(self.image_processor)} for argument '
                f'image_processor, but a {self.image_processor_class} '
                'was expected.')

        # TODO: support apply_ocr
        if self.image_processor.apply_ocr:
            raise ValueError(
                'Now only support initialized the image processor '
                'with apply_ocr set to False.')

        self.label_pad_token_id = label_pad_token_id

    def _resize_rescale_norm(self, results: dict) -> None:
        """apply the image_processor to process img."""
        img = results['img']
        h, w = results['img_shape']

        features: BatchFeature = self.image_processor(
            images=img, return_tensors='np', data_format=ChannelDimension.LAST)

        # output default dims NHWC and here N=1
        pixel_values = features['pixel_values'][0]
        new_h, new_w = pixel_values.shape[:2]
        w_scale = new_w / w
        h_scale = new_h / h
        results['pixel_values'] = pixel_values
        results['img_shape'] = (new_h, new_w)
        results['scale_factor'] = (w_scale, h_scale)

    def transform(self, results: dict) -> Dict:
        self._resize_rescale_norm(results)
        return results


@TRANSFORMS.register_module()
class ProcessTokenForLayoutLMv3(BaseTransform):
    """A transform to process token, which will dynamically pad the inputs
    received, as well as the labels.

    Part of code is modified from `https://github.com/microsoft/unilm/blob
    /master/layoutlmv3/layoutlmft/data/data_collator.py` and `https://
    github.com/huggingface/transformers/blob/main/src/transformers/models/
    layoutlmv3/processing_layoutlmv3.py`.

    Required Keys:

    - tokenizer
    - input_ids
    - attention_mask
    - labels
    - bbox
    - position_ids
    - segment_ids(optional)

    Modified Keys:

    - input_ids
    - attention_mask
    - labels
    - bbox
    - position_ids
    - segment_ids(optional)

    Args:
        padding (:obj:`bool`, :obj:`str` or :class:
        `~transformers.file_utils.PaddingStrategy`,
        `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences
            (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest
              sequence in the batch (or no padding if only a
              single sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified
              with the argument :obj:`max_length` or to the maximum
              acceptable input length for the model if that argument
              is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No
              padding (i.e., can output a batch with sequences
              of different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally
            padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the
            provided value. This is especially useful to enable the
            use of Tensor Cores on NVIDIA hardware with compute
            capability >= 7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be
            automatically ignore by PyTorch loss functions).
    """

    padded_input_names = ['input_ids', 'attention_mask']

    def __init__(self,
                 padding: Union[bool, str, PaddingStrategy] = True,
                 max_length: Optional[int] = None,
                 pad_to_multiple_of: Optional[int] = None,
                 label_pad_token_id: int = -100) -> None:
        super().__init__()
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id

    def _pad(self, results: dict) -> None:
        # get tokenizer
        tokenizer = results['tokenizer']

        # There will be a warning advice:
        # You're using a XLMRobertaTokenizerFast tokenizer.
        # Please note that with a fast tokenizer, using the
        # `__call__` method is faster than using a method to
        # encode the text followed by a call to the `pad`
        # method to get a padded encoding.
        # But `__call__` method only supports input string text,
        # which has already been encoded before this step.
        features = {
            k: v
            for k, v in results.items() if k in self.padded_input_names
        }
        batch: BatchEncoding = tokenizer.pad(
            encoded_inputs=features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of)
        # update `input_ids` and `attention_mask`
        results.update(batch)

        has_bbox_input = 'bbox' in results
        has_position_input = 'position_ids' in results
        has_segment_input = 'segment_ids' in results
        sequence_length = len(results['input_ids'])
        if tokenizer.padding_side == 'right':
            results[
                'labels'] = results['labels'] + [self.label_pad_token_id] * (
                    sequence_length - len(results['labels']))
            if has_bbox_input:
                results['bbox'] = results['bbox'] + [[0, 0, 0, 0]] * (
                    sequence_length - len(results['bbox']))
            if has_position_input:
                results['position_ids'] = results['position_ids'] + [
                    tokenizer.pad_token_id
                ] * (
                    sequence_length - len(results['position_ids']))
            if has_segment_input:
                results['segment_ids'] = results['segment_ids'] + [
                    results['segment_ids'][-1] + 1
                ] * (
                    sequence_length - len(results['segment_ids']))
        else:
            results['labels'] = [self.label_pad_token_id] * (
                sequence_length - len(results['labels'])) + results['labels']
            if has_bbox_input:
                results['bbox'] = [[0, 0, 0, 0]] * (
                    sequence_length - len(results['bbox'])) + results['bbox']
            if has_position_input:
                results['position_ids'] = [tokenizer.pad_token_id] * (
                    sequence_length -
                    len(results['position_ids'])) + results['position_ids']
            if has_segment_input:
                results['segment_ids'] = [results['segment_ids'][-1] + 1] * (
                    sequence_length -
                    len(results['segment_ids'])) + results['segment_ids']

    def transform(self, results: dict) -> Dict:
        self._pad(results)
        return results

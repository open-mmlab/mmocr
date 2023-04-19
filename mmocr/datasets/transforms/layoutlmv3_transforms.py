# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Union

from mmcv.transforms.base import BaseTransform

from mmocr.registry import TRANSFORMS
from transformers import LayoutLMv3ImageProcessor, LayoutXLMTokenizerFast
from transformers.file_utils import PaddingStrategy
from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import ChannelDimension
from transformers.tokenization_utils_base import (BatchEncoding,
                                                  TruncationStrategy)


@TRANSFORMS.register_module()
class LoadProcessorFromPretrainedModel(BaseTransform):
    """A transform to load image_processor/text_tokenizer from pretrained
    model, which will use HuggingFace `LayoutLMv3ImageProcessor` and
    `LayoutXLMTokenizerFast`

    Added Keys:

    - image_processor
    - tokeinzer

    Args:
        pretrained_model_name_or_path (str): The name or path of huggingface
            pretrained model, which must be specified.
        image_processor (dict): The specific parameters for image_processor.
        tokenizer (dict): The specific parameters for tokenizer.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        image_processor: dict = dict(),
        tokenizer: dict = dict()
    ) -> None:
        super().__init__()
        assert pretrained_model_name_or_path != ''
        self.image_processor = LayoutLMv3ImageProcessor.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            **image_processor)
        # TODO: support apply_ocr
        if self.image_processor.apply_ocr:
            raise ValueError(
                'Now only support initialized the image processor '
                'with apply_ocr set to False.')

        # https://huggingface.co/microsoft/layoutlmv3-base-chinese/discussions/3
        # use LayoutXLMTokenizerFast instead of LayoutLMv3TokenizerFast
        self.tokenizer = LayoutXLMTokenizerFast.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            **tokenizer)

    def transform(self, results: dict) -> Dict:
        results['image_processor'] = self.image_processor
        results['tokenizer'] = self.tokenizer
        return results


@TRANSFORMS.register_module()
class ProcessImageForLayoutLMv3(BaseTransform):
    """A transform to process image for LayoutLMv3.

    Required Keys:

    - img
    - img_shape
    - image_processor

    Modified Keys:

    - img_shape

    Added Keys:

    - scale_factor
    - pixel_values
    """

    def __init__(self) -> None:
        super().__init__()

    def _resize_rescale_norm(self, results: dict) -> None:
        """apply the image_processor to img."""
        img = results['img']
        h, w = results['img_shape']

        image_processor = results['image_processor']
        features: BatchFeature = image_processor(
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
    """A transform to process texts for LayoutLMv3,

    Required Keys:

    - tokenizer
    - width
    - height
    - instances
        - texts
        - boxes

    Added Keys:

    - input_ids
    - attention_mask
    - bbox
    - word_ids

    Args:
        Refer to the parameters of the corresponding tokenizer
    """

    def __init__(self,
                 padding: Union[bool, str, PaddingStrategy] = False,
                 max_length: Optional[int] = None,
                 truncation: Union[bool, str, TruncationStrategy] = None,
                 pad_to_multiple_of: Optional[int] = None) -> None:
        super().__init__()
        self.padding = padding
        self.max_length = max_length
        self.truncation = truncation
        self.pad_to_multiple_of = pad_to_multiple_of

    def _tokenize(self, results: dict) -> None:
        tokenizer = results['tokenizer']

        instances = results['instances']
        texts = instances['texts']
        boxes = instances['boxes']

        tokenized_inputs: BatchEncoding = tokenizer(
            text=texts,
            boxes=boxes,
            padding=self.padding,
            max_length=self.max_length,
            truncation=self.truncation,
            pad_to_multiple_of=self.pad_to_multiple_of,
            add_special_tokens=True,
            return_tensors='np',
            return_attention_mask=True,
            return_offsets_mapping=True)

        # By default, the pipeline processes one sample
        # at a time, so set batch_index = 0.
        batch_index = 0
        # record input_ids/attention_mask/bbox
        for k in ['input_ids', 'attention_mask', 'bbox']:
            results[k] = tokenized_inputs[k][batch_index]
        # record word_ids
        results['word_ids'] = tokenized_inputs.encodings[batch_index].word_ids

    def _norm_boxes(self, results: dict) -> None:

        def box_norm(box, width, height):

            def clip(min_num, num, max_num):
                return min(max(num, min_num), max_num)

            x0, y0, x1, y1 = box
            x0 = clip(0, int((x0 / width) * 1000), 1000)
            y0 = clip(0, int((y0 / height) * 1000), 1000)
            x1 = clip(0, int((x1 / width) * 1000), 1000)
            y1 = clip(0, int((y1 / height) * 1000), 1000)
            assert x1 >= x0
            assert y1 >= y0
            return [x0, y0, x1, y1]

        instances = results['instances']
        boxes = instances['boxes']

        # norm boxes
        width = results['width']
        height = results['height']
        norm_boxes = [box_norm(box, width, height) for box in boxes]

        results['instances']['boxes'] = norm_boxes

    def transform(self, results: dict) -> Dict:
        self._norm_boxes(results)
        self._tokenize(results)
        return results


@TRANSFORMS.register_module()
class ConvertBIOLabelForSER(BaseTransform):
    """A transform to convert BIO format labels for SER task,

    Required Keys:

    - tokenizer
    - word_ids
    - instances
        - labels

    Added Keys:

    - labels

    Args:
        classes (Union[tuple, list]): dataset classes
        only_label_first_subword (bool): Whether or not to only label
            the first subword, in case word labels are provided.
    """

    def __init__(self,
                 classes: Union[tuple, list],
                 only_label_first_subword: bool = False) -> None:
        super().__init__()
        self.biolabel2id = self._generate_biolabel2id_map(classes)
        self.only_label_first_subword = only_label_first_subword

    def _generate_biolabel2id_map(self, classes: Union[tuple, list]) -> Dict:
        bio_label_list = []
        classes = sorted([c.upper() for c in classes])
        for c in classes:
            if c == 'OTHER':
                bio_label_list.insert(0, c)
            else:
                bio_label_list.append(f'B-{c}')
                bio_label_list.append(f'I-{c}')
        biolabel2id_map = {
            bio_label: idx
            for idx, bio_label in enumerate(bio_label_list)
        }
        return biolabel2id_map

    def _convert(self, results: dict) -> None:
        tokenizer = results['tokenizer']

        instances = results['instances']
        labels = [label.upper() for label in instances['labels']]
        word_ids = results['word_ids']

        biolabel_ids = []
        pre_word_id = None
        for cur_word_id in word_ids:
            if cur_word_id is not None:
                if cur_word_id != pre_word_id:
                    biolabel_name = f'B-{labels[cur_word_id]}' \
                        if labels[cur_word_id] != 'OTHER' else 'OTHER'
                elif self.only_label_first_subword:
                    biolabel_name = 'OTHER'
                else:
                    biolabel_name = f'I-{labels[cur_word_id]}' \
                        if labels[cur_word_id] != 'OTHER' else 'OTHER'
                # convert biolabel to id
                biolabel_ids.append(self.biolabel2id[biolabel_name])
            else:
                biolabel_ids.append(tokenizer.pad_token_label)
            pre_word_id = cur_word_id

        # record biolabel_ids
        results['labels'] = biolabel_ids

    def transform(self, results: dict) -> Dict:
        self._convert(results)
        return results

# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Callable, List, Optional, Sequence, Union

from mmengine.dataset import BaseDataset
from transformers import AutoTokenizer

from mmocr.registry import DATASETS


@DATASETS.register_module()
class SERDataset(BaseDataset):

    def __init__(self,
                 ann_file: str = '',
                 tokenizer: str = '',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = '',
                 data_prefix: dict = dict(img_path=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000) -> None:

        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
        self.tokenizer = tokenizer

        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch)

    def load_data_list(self) -> List[dict]:
        data_list = super().load_data_list()

        # split text to several slices because of over-length
        split_text_data_list = []
        for i in range(len(data_list)):
            start = 0
            cur_iter = 0
            while start < len(data_list[i]['input_ids']):
                end = min(start + 510, len(data_list[i]['input_ids']))
                # get input_ids
                input_ids = [self.tokenizer.cls_token_id] + \
                    data_list[i]['input_ids'][start:end] + \
                    [self.tokenizer.sep_token_id]
                # get bboxes
                bboxes = [[0, 0, 0, 0]] + \
                    data_list[i]['bboxes'][start:end] + \
                    [[1000, 1000, 1000, 1000]]
                # get labels
                labels = [-100] + data_list[i]['labels'][start:end] + [-100]
                # get segment_ids
                segment_ids = self.get_segment_ids(bboxes)
                # get position_ids
                position_ids = self.get_position_ids(segment_ids)
                # get img_path
                img_path = os.path.join(self.data_root,
                                        data_list[i]['img_path'])
                # get attention_mask
                attention_mask = [1] * len(input_ids)

                data_info = {}
                data_info['input_ids'] = input_ids
                data_info['bboxes'] = bboxes
                data_info['labels'] = labels
                data_info['segment_ids'] = segment_ids
                data_info['position_ids'] = position_ids
                data_info['img_path'] = img_path
                data_info['attention_mask '] = attention_mask
                split_text_data_list.append(data_info)

                start = end
                cur_iter += 1

        return split_text_data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        instances = raw_data_info['instances']
        img_path = raw_data_info['img_path']
        width = raw_data_info['width']
        height = raw_data_info['height']

        texts = instances.get('texts', None)
        bboxes = instances.get('bboxes', None)
        labels = instances.get('labels', None)
        assert texts or bboxes or labels
        # norm box
        bboxes_norm = [self.box_norm(box, width, height) for box in bboxes]
        # get label2id
        label2id = self.metainfo['label2id']

        cur_doc_input_ids, cur_doc_bboxes, cur_doc_labels = [], [], []
        for j in range(len(texts)):
            cur_input_ids = self.tokenizer(
                texts[j],
                truncation=False,
                add_special_tokens=False,
                return_attention_mask=False)['input_ids']
            if len(cur_input_ids) == 0:
                continue

            cur_label = labels[j].upper()
            if cur_label == 'OTHER':
                cur_labels = ['O'] * len(cur_input_ids)
                for k in range(len(cur_labels)):
                    cur_labels[k] = label2id[cur_labels[k]]
            else:
                cur_labels = [cur_label] * len(cur_input_ids)
                cur_labels[0] = label2id['B-' + cur_labels[0]]
                for k in range(1, len(cur_labels)):
                    cur_labels[k] = label2id['I-' + cur_labels[k]]
            assert len(cur_input_ids) == len(
                [bboxes_norm[j]] * len(cur_input_ids)) == len(cur_labels)
            cur_doc_input_ids += cur_input_ids
            cur_doc_bboxes += [bboxes_norm[j]] * len(cur_input_ids)
            cur_doc_labels += cur_labels
        assert len(cur_doc_input_ids) == len(cur_doc_bboxes) == len(
            cur_doc_labels)
        assert len(cur_doc_input_ids) > 0

        data_info = {}
        data_info['img_path'] = img_path
        data_info['input_ids'] = cur_doc_input_ids
        data_info['bboxes'] = cur_doc_bboxes
        data_info['labels'] = cur_doc_labels
        return data_info

    def box_norm(self, box, width, height):

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

    def get_segment_ids(self, bboxs):
        segment_ids = []
        for i in range(len(bboxs)):
            if i == 0:
                segment_ids.append(0)
            else:
                if bboxs[i - 1] == bboxs[i]:
                    segment_ids.append(segment_ids[-1])
                else:
                    segment_ids.append(segment_ids[-1] + 1)
        return segment_ids

    def get_position_ids(self, segment_ids):
        position_ids = []
        for i in range(len(segment_ids)):
            if i == 0:
                position_ids.append(2)
            else:
                if segment_ids[i] == segment_ids[i - 1]:
                    position_ids.append(position_ids[-1] + 1)
                else:
                    position_ids.append(2)
        return position_ids

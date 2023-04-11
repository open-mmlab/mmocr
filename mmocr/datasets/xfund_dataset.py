# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Callable, List, Optional, Sequence, Union

from mmengine.dataset import BaseDataset

from mmocr.registry import DATASETS
from transformers import AutoTokenizer


@DATASETS.register_module()
class XFUNDSERDataset(BaseDataset):
    """XFUND Dataset for Semantic Entity Recognition task. part of code is
    modified from https://github.com/microsoft/unilm/blob/master/layoutlmv3/lay
    outlmft/data/xfund.py.

    Args:
        ann_file (str): Annotation file path. Defaults to ''.
        tokenizer (str): The pre-trained tokenizer you want to use.
            Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        data_prefix (dict): Prefix for training data. Defaults to
            ``dict(img_path='')``.
        filter_cfg (dict, optional): Config for filter data. Defaults to None.
        indices (int or Sequence[int], optional): Support using first few
            data in annotation file to facilitate training/testing on a smaller
            dataset. Defaults to None which means using all ``data_infos``.
        serialize_data (bool, optional): Whether to hold memory using
            serialized objects, when enabled, data loader workers can use
            shared RAM from master process instead of making a copy. Defaults
            to True.
        pipeline (list, optional): Processing pipeline. Defaults to [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Defaults to False.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. In some cases, such as visualization, only the meta
            information of the dataset is needed, which is not necessary to
            load annotation file. ``RecogLMDBDataset`` can skip load
            annotations to save time by set ``lazy_init=False``.
            Defaults to False.
        max_refetch (int, optional): If ``RecogLMDBdataset.prepare_data`` get a
            None img. The maximum extra number of cycles to get a valid
            image. Defaults to 1000.
    """

    def __init__(self,
                 ann_file: str,
                 tokenizer: dict,
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

        if isinstance(tokenizer, dict) and \
                tokenizer.get('pretrained_model_name_or_path', None):
            self.tokenizer = AutoTokenizer.from_pretrained(**tokenizer)
        else:
            raise TypeError(
                'tokenizer cfg should be a `dict` and a key '
                '`pretrained_model_name_or_path` must be specified')

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
                # get boxes
                boxes = [[0, 0, 0, 0]] + \
                    data_list[i]['boxes'][start:end] + \
                    [[1000, 1000, 1000, 1000]]
                # get labels
                labels = [-100] + data_list[i]['labels'][start:end] + [-100]
                # get segment_ids
                segment_ids = self.get_segment_ids(boxes)
                # get position_ids
                position_ids = self.get_position_ids(segment_ids)
                # get img_path
                img_path = os.path.join(self.data_root,
                                        data_list[i]['img_path'])
                # get attention_mask
                attention_mask = [1] * len(input_ids)

                data_info = {}
                data_info['input_ids'] = input_ids
                data_info['boxes'] = boxes
                data_info['labels'] = labels
                data_info['segment_ids'] = segment_ids
                data_info['position_ids'] = position_ids
                data_info['img_path'] = img_path
                data_info['attention_mask'] = attention_mask
                # record biolabel2id and id2biolabel
                biolabel2id = self.metainfo['biolabel2id']
                data_info['biolabel2id'] = biolabel2id
                id2biolabel = {v: k for k, v in biolabel2id.items()}
                data_info['id2biolabel'] = id2biolabel
                split_text_data_list.append(data_info)

                start = end
                cur_iter += 1

        return split_text_data_list

    def parse_data_info(self, raw_data_info: dict) -> dict:
        """Parse raw data information, tokenize texts and normalize boxes.

        raw_data_info
                    {
                        "img_path": "imgs\\test\\zh_val_0.jpg",
                        "height": 3508,
                        "width": 2480,
                        "instances":
                        {
                            "texts": ["汇丰晋信", "受理时间:", ...],
                            "boxes": [[104, 114, 530, 175],
                                      [126, 267, 266, 305], ...],
                            "labels": ["other", "question", ...],
                            "words": [[...], [...], ...]
                        }
                    }
        will be modified to data_info
                    {
                        "img_path": "imgs\\test\\zh_val_0.jpg",
                        "input_ids": [6, 47360, 49222, 124321, 5070, 6, ...],
                        "boxes": [[41, 32, 213, 49],
                                  [41, 32, 213, 49],
                                  [41, 32, 213, 49],
                                  [41, 32, 213, 49],
                                  [41, 32, 213, 49],
                                  [50, 76, 107, 86], ...],
                        "labels": [0, 0, 0, 0, 0, 1, ...]
                    }
        The length of `texts`、`boxes` and `labels` will increase.
        The `words` annotations are not used here.
        """
        instances = raw_data_info['instances']
        texts = instances['texts']
        boxes = instances['boxes']
        labels = instances['labels']

        # norm boxes
        width = raw_data_info['width']
        height = raw_data_info['height']
        norm_boxes = [self.box_norm(box, width, height) for box in boxes]

        # get biolabel2id
        biolabel2id = self.metainfo['biolabel2id']
        # tokenize texts
        cur_doc_input_ids, cur_doc_boxes, cur_doc_labels = [], [], []
        for j in range(len(texts)):
            cur_input_ids = self.tokenizer(
                texts[j],
                truncation=False,
                add_special_tokens=False,
                return_attention_mask=False)['input_ids']
            if len(cur_input_ids) == 0:
                continue
            # generate bio label
            cur_label = labels[j].upper()
            if cur_label == 'OTHER':
                cur_labels = ['O'] * len(cur_input_ids)
                for k in range(len(cur_labels)):
                    cur_labels[k] = biolabel2id[cur_labels[k]]
            else:
                cur_labels = [cur_label] * len(cur_input_ids)
                cur_labels[0] = biolabel2id['B-' + cur_labels[0]]
                for k in range(1, len(cur_labels)):
                    cur_labels[k] = biolabel2id['I-' + cur_labels[k]]
            assert len(cur_input_ids) == len(cur_labels)

            cur_doc_input_ids += cur_input_ids
            cur_doc_boxes += [norm_boxes[j]] * len(cur_input_ids)
            cur_doc_labels += cur_labels
        assert len(cur_doc_input_ids) == len(cur_doc_boxes) == len(
            cur_doc_labels)
        assert len(cur_doc_input_ids) > 0

        data_info = {}
        data_info['img_path'] = raw_data_info['img_path']
        data_info['input_ids'] = cur_doc_input_ids
        data_info['boxes'] = cur_doc_boxes
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

    def get_segment_ids(self, boxes):
        segment_ids = []
        for i in range(len(boxes)):
            if i == 0:
                segment_ids.append(0)
            else:
                if boxes[i - 1] == boxes[i]:
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

# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Dict, Tuple

import mmcv

from mmocr.registry import DATA_PACKERS
from .ser_packer import SERPacker


@DATA_PACKERS.register_module()
class REPacker(SERPacker):
    """Relation Extraction packer. It is used to pack the parsed annotation
    info to.

    .. code-block:: python

        {
            "metainfo":
                {
                    "labels": ['answer', 'header', 'other', 'question'],
                    "id2label": {
                        "0": "O",
                        "1": "B-ANSWER",
                        "2": "I-ANSWER",
                        "3": "B-HEADER",
                        "4": "I-HEADER",
                        "5": "B-QUESTION",
                        "6": "I-QUESTION"
                    },
                    "label2id": {
                        "O": 0,
                        "B-ANSWER": 1,
                        "I-ANSWER": 2,
                        "B-HEADER": 3,
                        "I-HEADER": 4,
                        "B-QUESTION": 5,
                        "I-QUESTION": 6
                    }
                },
            "data_list":
                [
                    {
                        "img_path": "ser_imgs\\test\\zh_val_0.jpg",
                        "height": 3508,
                        "width": 2480,
                        "instances":
                        {
                            "texts": ["绩效目标申报表(一级项目)", "项目名称", ...],
                            "boxes": [[906,195,1478,259],
                                      [357,325,467,357], ...],
                            "labels": ["header", "question", ...],
                            "linkings": [[0, 1], [2, 3], ...],
                            "ids": [0, 1, ...],
                            "words": [[{
                                        "box": [
                                            904,
                                            192,
                                            942,
                                            253
                                        ],
                                        "text": "绩"
                                    },
                                    {
                                        "box": [
                                            953,
                                            192,
                                            987,
                                            253
                                        ],
                                        "text": "效"
                                    }, ...], ...]
                        }
                    }
                ]
        }
    """

    def pack_instance(self, sample: Tuple) -> Dict:
        """Pack the parsed annotation info to an MMOCR format instance.

        Args:
            sample (Tuple): A tuple of (img_file, instances).
               - img_path (str): Path to the image file.
               - instances (Sequence[Dict]): A list of converted annos. Each
                 element should be a dict with the following keys:

                 - 'text'
                 - 'box'
                 - 'label'
                 - 'linking'
                 - 'id'
                 - 'words' (optional)

        Returns:
            Dict: An MMOCR format instance.
        """

        img_path, instances = sample

        img = mmcv.imread(img_path)
        h, w = img.shape[:2]

        texts_per_doc = []
        boxes_per_doc = []
        labels_per_doc = []
        linking_per_doc = []
        id_per_doc = []
        has_words = all(['words' in ins for ins in instances])
        if has_words:
            words_per_doc = []
        else:
            warnings.warn(
                'Not all instance has `words` key,'
                'so final MMOCR format SER instance will not have `words` key')

        for instance in instances:
            text = instance.get('text', None)
            box = instance.get('box', None)
            label = instance.get('label', None)
            linking = instance.get('linking', None)
            ins_id = instance.get('id', None)
            assert text or box or label or linking or ins_id
            texts_per_doc.append(text)
            boxes_per_doc.append(box)
            labels_per_doc.append(label)
            linking_per_doc.append(linking)
            id_per_doc.append(ins_id)
            if has_words:
                words = instance.get('words', None)
                words_per_doc.append(words)
        packed_instances = dict(
            instances=dict(
                texts=texts_per_doc,
                boxes=boxes_per_doc,
                labels=labels_per_doc,
                linkings=linking_per_doc,
                ids=id_per_doc),
            img_path=osp.relpath(img_path, self.data_root),
            height=h,
            width=w)
        if has_words:
            packed_instances['instances'].update({'words': words_per_doc})

        return packed_instances

# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Dict, List, Tuple

import mmcv

from mmocr.registry import DATA_PACKERS
from .base import BasePacker


@DATA_PACKERS.register_module()
class REPacker(BasePacker):
    """Relation Extraction packer. It is used to pack the parsed annotation
    info to.

    .. code-block:: python

        {
            "metainfo":
                {
                    "dataset_type": "REDataset",
                    "task_name": "re",
                    "labels": ['answer', 'header', 'other', 'question'],
                    "id2label": {
                        "0": "answer",
                        "1": "header",
                        "2": "other",
                        "3": "question"
                    },
                    "label2id": {
                        "answer": 0,
                        "header": 1,
                        "other": 2,
                        "question": 3
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
                            "bboxes": [[906,195,1478,259],
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
        bboxes_per_doc = []
        labels_per_doc = []
        words_per_doc = []
        linking_per_doc = []
        id_per_doc = []
        for instance in instances:
            text = instance.get('text', None)
            box = instance.get('box', None)
            label = instance.get('label', None)
            linking = instance.get('linking', None)
            ins_id = instance.get('id', None)
            words = instance.get('words', None)
            assert text or box or label
            texts_per_doc.append(text)
            bboxes_per_doc.append(box)
            labels_per_doc.append(label)
            words_per_doc.append(words)
            linking_per_doc.append(linking)
            id_per_doc.append(ins_id)
        packed_instances = dict(
            instances=dict(
                texts=texts_per_doc,
                bboxes=bboxes_per_doc,
                labels=labels_per_doc,
                linkings=linking_per_doc,
                ids=id_per_doc,
                words=words_per_doc),
            img_path=osp.relpath(img_path, self.data_root),
            height=h,
            width=w)

        return packed_instances

    def add_meta(self, sample: List) -> Dict:
        """Add meta information to the sample.

        Args:
            sample (List): A list of samples of the dataset.

        Returns:
            Dict: A dict contains the meta information and samples.
        """

        labels = []
        for s in sample:
            labels += s['instances']['labels']
        label_list = list(set(labels))
        label_list.sort()

        meta = {
            'metainfo': {
                'dataset_type': 'REDataset',
                'task_name': 're',
                'labels': label_list,
                'id2label': {k: v
                             for k, v in enumerate(label_list)},
                'label2id': {v: k
                             for k, v in enumerate(label_list)}
            },
            'data_list': sample
        }
        return meta

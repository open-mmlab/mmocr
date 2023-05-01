# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset import BaseDataset

from mmocr.registry import DATASETS


@DATASETS.register_module()
class XFUNDDataset(BaseDataset):
    """XFUND Dataset for Semantic Entity Recognition and Relation Extraction
    task.

    The annotation format is shown as follows.

    .. code-block:: none

        {
            "metainfo":{},
            "data_list":
            [
              {
                "img_path": "data/xfund/zh/imgs/train/zh_train_0.jpg",
                "height": 3508,
                "width": 2480,
                "instances":
                {
                    "texts": ["绩效目标申报表(一级项目)", "项目名称", ...],
                    "boxes": [[906,195,1478,259],
                                [357,325,467,357], ...],
                    "labels": ["header", "question", ...],
                    "linkings": [[0, 1], [2, 3], ...], (RE task will have)
                    "ids": [0, 1, ...], (RE task will have)
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
              },
            ]
        }

    Args:
        The same as OCRDataset
    """

# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, Sequence

import mmcv

from mmocr.utils import is_type_list


def dump_ocr_data(image_infos: Sequence[Dict], out_json_name: str,
                  task_name: str, **kwargs) -> Dict:
    """Dump the annotation in openmmlab style.

    Args:
        image_infos (list): List of image information dicts. Read the example
            section for the format illustration.
        out_json_name (str): Output json filename.
        task_name (str): Task name. Options are 'textdet', 'textrecog' and
            'textspotter'.

    Examples:
        Here is the general structure of image_infos for textdet/textspotter
        tasks:

        .. code-block:: python

            [  # A list of dicts. Each dict stands for a single image.
                {
                    "file_name": "1.jpg",
                    "height": 100,
                    "width": 200,
                    "segm_file": "seg.txt" # (optional) path to segmap
                    "anno_info": [  # a list of dicts. Each dict
                                    # stands for a single text instance.
                        {
                            "iscrowd": 0,  # 0: don't ignore this instance
                                           # 1: ignore
                            "category_id": 0,  # Instance class id. Must be 0
                                               # for OCR tasks to permanently
                                               # be mapped to 'text' category
                            "bbox": [x, y, w, h],
                            "segmentation": [x1, y1, x2, y2, ...],
                            "text": "demo_text"  # for textspotter only.
                        }
                    ]
                },
            ]

        The input for textrecog task is much simpler:

        .. code-block:: python

            [   # A list of dicts. Each dict stands for a single image.
                {
                    "file_name": "1.jpg",
                    "anno_info": [  # a list of dicts. Each dict
                                    # stands for a single text instance.
                                    # However, in textrecog, usually each
                                    # image only has one text instance.
                        {
                            "text": "demo_text"
                        }
                    ]
                },
            ]


    Returns:
        out_json(dict): The openmmlab-style annotation.
    """

    task2dataset = {
        'textspotter': 'TextSpotterDataset',
        'textdet': 'TextDetDataset',
        'textrecog': 'TextRecogDataset'
    }

    assert isinstance(image_infos, list)
    assert isinstance(out_json_name, str)
    assert task_name in task2dataset.keys()

    dataset_type = task2dataset[task_name]

    out_json = dict(
        metainfo=dict(dataset_type=dataset_type, task_name=task_name),
        data_list=list())
    if task_name in ['textdet', 'textspotter']:
        out_json['metainfo']['category'] = [dict(id=0, name='text')]

    for image_info in image_infos:

        single_info = dict(instances=list())
        single_info['img_path'] = image_info['file_name']
        if task_name in ['textdet', 'textspotter']:
            single_info['height'] = image_info['height']
            single_info['width'] = image_info['width']
            if 'segm_file' in image_info:
                single_info['seg_map'] = image_info['segm_file']

        anno_infos = image_info['anno_info']

        for anno_info in anno_infos:
            instance = {}
            if task_name in ['textrecog', 'textspotter']:
                instance['text'] = anno_info['text']
            if task_name in ['textdet', 'textspotter']:
                mask = anno_info['segmentation']
                # TODO: remove this if-branch when all converters have been
                # verified
                if len(mask) == 1 and len(mask[0]) > 1:
                    mask = mask[0]
                    warnings.warn(
                        'Detected nested segmentation for a single'
                        'text instance, which should be a 1-d array now.'
                        'Please fix input accordingly.')
                instance['polygon'] = mask
                x, y, w, h = anno_info['bbox']
                instance['bbox'] = [x, y, x + w, y + h]
                instance['bbox_label'] = anno_info['category_id']
                instance['ignore'] = anno_info['iscrowd'] == 1
            single_info['instances'].append(instance)

        out_json['data_list'].append(single_info)

    mmcv.dump(out_json, out_json_name, **kwargs)

    return out_json


def recog_anno_to_imginfo(
    file_paths: Sequence[str],
    labels: Sequence[str],
) -> Sequence[Dict]:
    """Convert a list of file_paths and labels for recognition tasks into the
    format of image_infos acceptable by :func:`dump_ocr_data()`. It's meant to
    maintain compatibility with the legacy annotation format in MMOCR 0.x.

    In MMOCR 0.x, data converters for recognition usually converts the
    annotations into a list of file paths and a list of labels, which look
    like the following:

    .. code-block:: python

        file_paths = ['1.jpg', '2.jpg', ...]
        labels = ['aaa', 'bbb', ...]

    This utility merges them into a list of dictionaries parsable by
    :func:`dump_ocr_data()`:

    .. code-block:: python

        [   # A list of dicts. Each dict stands for a single image.
            {
                "file_name": "1.jpg",
                "anno_info": [
                    {
                        "text": "aaa"
                    }
                ]
            },
            {
                "file_name": "2.jpg",
                "anno_info": [
                    {
                        "text": "bbb"
                    }
                ]
            },
            ...
        ]

    Args:
        file_paths (list[str]): A list of file paths to images.
        labels (list[str]): A list of text labels.

    Returns:
        list[dict]: Annotations parsable by :func:`dump_ocr_data()`.
    """
    assert is_type_list(file_paths, str)
    assert is_type_list(labels, str)
    assert len(file_paths) == len(labels)

    results = []
    for i in range(len(file_paths)):
        result = dict(
            file_name=file_paths[i], anno_info=[dict(text=labels[i])])
        results.append(result)

    return results

# Copyright (c) OpenMMLab. All rights reserved.
import json
import math
import os.path as osp
import tempfile

import torch

from mmocr.datasets.openset_kie_dataset import OpensetKIEDataset
from mmocr.utils import list_to_file


def _create_dummy_ann_file(ann_file):
    ann_info1 = {
        'file_name':
        '1.png',
        'height':
        200,
        'width':
        200,
        'annotations': [{
            'text': 'store',
            'box': [11.0, 0.0, 22.0, 0.0, 12.0, 12.0, 0.0, 12.0],
            'label': 1,
            'edge': 1
        }, {
            'text': 'MyFamily',
            'box': [23.0, 2.0, 31.0, 1.0, 24.0, 11.0, 16.0, 11.0],
            'label': 2,
            'edge': 1
        }]
    }
    list_to_file(ann_file, [json.dumps(ann_info1)])

    return ann_info1


def _create_dummy_dict_file(dict_file):
    dict_str = '0123'
    list_to_file(dict_file, list(dict_str))


def _create_dummy_loader():
    loader = dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineJsonParser',
            keys=['file_name', 'height', 'width', 'annotations']))
    return loader


def test_openset_kie_dataset():
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        # create dummy data
        ann_file = osp.join(tmp_dir_name, 'fake_data.txt')
        ann_info1 = _create_dummy_ann_file(ann_file)

        dict_file = osp.join(tmp_dir_name, 'fake_dict.txt')
        _create_dummy_dict_file(dict_file)

        # test initialization
        loader = _create_dummy_loader()
        dataset = OpensetKIEDataset(ann_file, loader, dict_file, pipeline=[])

        dataset.prepare_train_img(0)

        # test pre_pipeline
        img_ann_info = dataset.data_infos[0]
        img_info = {
            'filename': img_ann_info['file_name'],
            'height': img_ann_info['height'],
            'width': img_ann_info['width']
        }
        ann_info = dataset._parse_anno_info(img_ann_info['annotations'])
        results = dict(img_info=img_info, ann_info=ann_info)
        dataset.pre_pipeline(results)
        assert results['img_prefix'] == dataset.img_prefix
        assert 'ori_texts' in results

        # test evaluation
        result = {
            'img_metas': [{
                'filename': ann_info1['file_name'],
                'ori_filename': ann_info1['file_name'],
                'ori_texts': [],
                'ori_boxes': []
            }]
        }
        for anno in ann_info1['annotations']:
            result['img_metas'][0]['ori_texts'].append(anno['text'])
            result['img_metas'][0]['ori_boxes'].append(anno['box'])
        result['nodes'] = torch.tensor([[0.01, 0.8, 0.01, 0.18],
                                        [0.01, 0.01, 0.9, 0.08]])
        result['edges'] = torch.Tensor([[0.01, 0.99] for _ in range(4)])

        eval_res = dataset.evaluate([result])
        assert math.isclose(eval_res['edge_openset_f1'], 1.0, abs_tol=1e-4)

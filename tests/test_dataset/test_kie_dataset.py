import json
import math
import os.path as osp
import tempfile

import pytest
import torch

from mmocr.datasets.kie_dataset import KIEDataset


def _create_dummy_ann_file(ann_file):
    ann_info1 = {
        'file_name':
        'sample1.png',
        'height':
        200,
        'width':
        200,
        'annotations': [{
            'text': 'store',
            'box': [11.0, 0.0, 22.0, 0.0, 12.0, 12.0, 0.0, 12.0],
            'label': 1
        }, {
            'text': 'address',
            'box': [23.0, 2.0, 31.0, 1.0, 24.0, 11.0, 16.0, 11.0],
            'label': 1
        }, {
            'text': 'price',
            'box': [33.0, 2.0, 43.0, 2.0, 36.0, 12.0, 25.0, 12.0],
            'label': 1
        }, {
            'text': '1.0',
            'box': [46.0, 2.0, 61.0, 2.0, 53.0, 12.0, 39.0, 12.0],
            'label': 1
        }, {
            'text': 'google',
            'box': [61.0, 2.0, 69.0, 2.0, 63.0, 12.0, 55.0, 12.0],
            'label': 1
        }]
    }
    with open(ann_file, 'w') as fw:
        for ann_info in [ann_info1]:
            fw.write(json.dumps(ann_info) + '\n')

    return ann_info1


def _create_dummy_dict_file(dict_file):
    dict_str = '0123'
    with open(dict_file, 'w') as fw:
        for char in list(dict_str):
            fw.write(char + '\n')

    return dict_str


def _create_dummy_loader():
    loader = dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineJsonParser',
            keys=['file_name', 'height', 'width', 'annotations']))
    return loader


def test_kie_dataset():
    tmp_dir = tempfile.TemporaryDirectory()
    # create dummy data
    ann_file = osp.join(tmp_dir.name, 'fake_data.txt')
    ann_info1 = _create_dummy_ann_file(ann_file)

    dict_file = osp.join(tmp_dir.name, 'fake_dict.txt')
    _create_dummy_dict_file(dict_file)

    # test initialization
    loader = _create_dummy_loader()
    dataset = KIEDataset(ann_file, loader, dict_file, pipeline=[])

    tmp_dir.cleanup()

    # test pre_pipeline
    img_info = dataset.data_infos[0]
    results = dict(img_info=img_info)
    dataset.pre_pipeline(results)
    assert results['img_prefix'] == dataset.img_prefix

    # test _parse_anno_info
    annos = ann_info1['annotations']
    with pytest.raises(AssertionError):
        dataset._parse_anno_info(annos[0])
    tmp_annos = [{
        'text': 'store',
        'box': [11.0, 0.0, 22.0, 0.0, 12.0, 12.0, 0.0, 12.0]
    }]
    dataset._parse_anno_info(tmp_annos)
    tmp_annos = [{'text': 'store'}]
    with pytest.raises(AssertionError):
        dataset._parse_anno_info(tmp_annos)

    return_anno = dataset._parse_anno_info(annos)
    assert 'bboxes' in return_anno
    assert 'relations' in return_anno
    assert 'texts' in return_anno
    assert 'labels' in return_anno

    # test evaluation
    result = {}
    result['nodes'] = torch.full((5, 5), 1, dtype=torch.float)
    result['nodes'][:, 1] = 100.
    print('hello', result['nodes'].size())
    results = [result for _ in range(5)]

    eval_res = dataset.evaluate(results)
    assert math.isclose(eval_res['macro_f1'], 0.2, abs_tol=1e-4)

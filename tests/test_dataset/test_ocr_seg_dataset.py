import json
import math
import os.path as osp
import tempfile

import pytest

from mmocr.datasets.ocr_seg_dataset import OCRSegDataset


def _create_dummy_ann_file(ann_file):
    ann_info1 = {
        'file_name':
        'sample1.png',
        'annotations': [{
            'char_text':
            'F',
            'char_box': [11.0, 0.0, 22.0, 0.0, 12.0, 12.0, 0.0, 12.0]
        }, {
            'char_text':
            'r',
            'char_box': [23.0, 2.0, 31.0, 1.0, 24.0, 11.0, 16.0, 11.0]
        }, {
            'char_text':
            'o',
            'char_box': [33.0, 2.0, 43.0, 2.0, 36.0, 12.0, 25.0, 12.0]
        }, {
            'char_text':
            'm',
            'char_box': [46.0, 2.0, 61.0, 2.0, 53.0, 12.0, 39.0, 12.0]
        }, {
            'char_text':
            ':',
            'char_box': [61.0, 2.0, 69.0, 2.0, 63.0, 12.0, 55.0, 12.0]
        }],
        'text':
        'From:'
    }
    ann_info2 = {
        'file_name':
        'sample2.png',
        'annotations': [{
            'char_text': 'o',
            'char_box': [0.0, 5.0, 7.0, 5.0, 9.0, 15.0, 2.0, 15.0]
        }, {
            'char_text':
            'u',
            'char_box': [7.0, 4.0, 14.0, 4.0, 18.0, 18.0, 11.0, 18.0]
        }, {
            'char_text':
            't',
            'char_box': [13.0, 1.0, 19.0, 2.0, 24.0, 18.0, 17.0, 18.0]
        }],
        'text':
        'out'
    }

    with open(ann_file, 'w') as fw:
        for ann_info in [ann_info1, ann_info2]:
            fw.write(json.dumps(ann_info) + '\n')

    return ann_info1, ann_info2


def _create_dummy_loader():
    loader = dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineJsonParser', keys=['file_name', 'text', 'annotations']))
    return loader


def test_ocr_seg_dataset():
    tmp_dir = tempfile.TemporaryDirectory()
    # create dummy data
    ann_file = osp.join(tmp_dir.name, 'fake_data.txt')
    ann_info1, ann_info2 = _create_dummy_ann_file(ann_file)

    # test initialization
    loader = _create_dummy_loader()
    dataset = OCRSegDataset(ann_file, loader, pipeline=[])

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
    annos2 = ann_info2['annotations']
    with pytest.raises(AssertionError):
        dataset._parse_anno_info([{'char_text': 'i'}])
    with pytest.raises(AssertionError):
        dataset._parse_anno_info([{'char_box': [1, 2, 3, 4, 5, 6, 7, 8]}])
    annos2[0]['char_box'] = [1, 2, 3]
    with pytest.raises(AssertionError):
        dataset._parse_anno_info(annos2)

    return_anno = dataset._parse_anno_info(annos)
    assert return_anno['chars'] == ['F', 'r', 'o', 'm', ':']
    assert len(return_anno['char_rects']) == 5

    # test prepare_train_img
    expect_results = {
        'img_info': {
            'filename': 'sample1.png'
        },
        'img_prefix': '',
        'ann_info': return_anno
    }
    data = dataset.prepare_train_img(0)
    assert data == expect_results

    # test evluation
    metric = 'acc'
    results = [{'text': 'From:'}, {'text': 'ou'}]
    eval_res = dataset.evaluate(results, metric)

    assert math.isclose(eval_res['word_acc'], 0.5, abs_tol=1e-4)
    assert math.isclose(eval_res['char_precision'], 1.0, abs_tol=1e-4)
    assert math.isclose(eval_res['char_recall'], 0.857, abs_tol=1e-4)

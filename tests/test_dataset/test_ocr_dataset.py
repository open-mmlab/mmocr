# Copyright (c) OpenMMLab. All rights reserved.
import math
import os.path as osp
import tempfile

from mmocr.datasets.ocr_dataset import OCRDataset


def _create_dummy_ann_file(ann_file):
    ann_info1 = 'sample1.jpg hello'
    ann_info2 = 'sample2.jpg world'

    with open(ann_file, 'w') as fw:
        for ann_info in [ann_info1, ann_info2]:
            fw.write(ann_info + '\n')


def _create_dummy_loader():
    loader = dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(type='LineStrParser', keys=['file_name', 'text']))
    return loader


def test_detect_dataset():
    tmp_dir = tempfile.TemporaryDirectory()
    # create dummy data
    ann_file = osp.join(tmp_dir.name, 'fake_data.txt')
    _create_dummy_ann_file(ann_file)

    # test initialization
    loader = _create_dummy_loader()
    dataset = OCRDataset(ann_file, loader, pipeline=[])

    tmp_dir.cleanup()

    # test pre_pipeline
    img_info = dataset.data_infos[0]
    results = dict(img_info=img_info)
    dataset.pre_pipeline(results)
    assert results['img_prefix'] == dataset.img_prefix
    assert results['text'] == img_info['text']

    # test evluation
    metric = 'acc'
    results = [{'text': 'hello'}, {'text': 'worl'}]
    eval_res = dataset.evaluate(results, metric)

    assert math.isclose(eval_res['word_acc'], 0.5, abs_tol=1e-4)
    assert math.isclose(eval_res['char_precision'], 1.0, abs_tol=1e-4)
    assert math.isclose(eval_res['char_recall'], 0.9, abs_tol=1e-4)

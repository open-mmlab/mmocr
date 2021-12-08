# Copyright (c) OpenMMLab. All rights reserved.
import copy

from mmocr.datasets import UniformConcatDataset
from mmocr.utils import list_from_file


def test_dataset_warpper():
    pipeline1 = [dict(type='LoadImageFromFile')]
    pipeline2 = [dict(type='LoadImageFromFile'), dict(type='ColorJitter')]

    img_prefix = 'tests/data/ocr_toy_dataset/imgs'
    ann_file = 'tests/data/ocr_toy_dataset/label.txt'
    train1 = dict(
        type='OCRDataset',
        img_prefix=img_prefix,
        ann_file=ann_file,
        loader=dict(
            type='HardDiskLoader',
            repeat=1,
            parser=dict(
                type='LineStrParser',
                keys=['filename', 'text'],
                keys_idx=[0, 1],
                separator=' ')),
        pipeline=None,
        test_mode=False)

    train2 = {key: value for key, value in train1.items()}
    train2['pipeline'] = pipeline2

    # pipeline is 1d list
    copy_train1 = copy.deepcopy(train1)
    copy_train2 = copy.deepcopy(train2)
    tmp_dataset = UniformConcatDataset(
        datasets=[copy_train1, copy_train2],
        pipeline=pipeline1,
        force_apply=True)

    assert len(tmp_dataset) == 2 * len(list_from_file(ann_file))
    assert len(tmp_dataset.datasets[0].pipeline.transforms) == len(
        tmp_dataset.datasets[1].pipeline.transforms)

    # pipeline is None
    copy_train2 = copy.deepcopy(train2)
    tmp_dataset = UniformConcatDataset(datasets=[copy_train2], pipeline=None)
    assert len(tmp_dataset.datasets[0].pipeline.transforms) == len(pipeline2)

    copy_train2 = copy.deepcopy(train2)
    tmp_dataset = UniformConcatDataset(
        datasets=[[copy_train2], [copy_train2]], pipeline=None)
    assert len(tmp_dataset.datasets[0].pipeline.transforms) == len(pipeline2)

    # pipeline is 2d list
    copy_train1 = copy.deepcopy(train1)
    copy_train2 = copy.deepcopy(train2)
    tmp_dataset = UniformConcatDataset(
        datasets=[[copy_train1], [copy_train2]],
        pipeline=[pipeline1, pipeline2])
    assert len(tmp_dataset.datasets[0].pipeline.transforms) == len(pipeline1)

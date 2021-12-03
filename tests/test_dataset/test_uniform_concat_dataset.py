# Copyright (c) OpenMMLab. All rights reserved.
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

    # test pipeline is 1d list
    uniform_concat_dataset = UniformConcatDataset(
        datasets=[train1, train2], pipeline=pipeline1)

    assert len(uniform_concat_dataset) == 2 * len(list_from_file(ann_file))
    assert len(uniform_concat_dataset.datasets[0].pipeline.transforms) != len(
        uniform_concat_dataset.datasets[1].pipeline.transforms)

    # test pipeline is None
    tmp_dataset = UniformConcatDataset(datasets=[train2], pipeline=None)
    assert len(tmp_dataset.datasets[0].pipeline.transforms) == len(pipeline2)

    # test pipeline is 2d list
    tmp_dataset = UniformConcatDataset(
        datasets=[train1, train2], pipeline=[pipeline1, pipeline2])
    assert len(tmp_dataset.datasets[0].pipeline.transforms) == len(pipeline1)

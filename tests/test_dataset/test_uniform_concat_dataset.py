# Copyright (c) OpenMMLab. All rights reserved.
import copy

import pytest
from mmdet.datasets import DATASETS

from mmocr.datasets import UniformConcatDataset
from mmocr.utils import list_from_file


def test_uniform_concat_dataset_pipeline():
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


def test_uniform_concat_dataset_eval():

    @DATASETS.register_module()
    class DummyDataset:

        def __init__(self):
            self.CLASSES = 0
            self.ann_file = 'empty'

        def __len__(self):
            return 1

        def evaluate(self, res, logger, **kwargs):
            return dict(n=res[0])

    fake_inputs = [10, 20]
    datasets = [dict(type='DummyDataset'), dict(type='DummyDataset')]

    tmp_dataset = UniformConcatDataset(datasets)
    results = tmp_dataset.evaluate(fake_inputs)
    assert results['0_n'] == 10
    assert results['1_n'] == 20

    tmp_dataset = UniformConcatDataset(datasets, show_mean_scores=True)
    results = tmp_dataset.evaluate(fake_inputs)
    assert results['0_n'] == 10
    assert results['1_n'] == 20
    assert results['mean_n'] == 15

    with pytest.raises(NotImplementedError):
        ds = UniformConcatDataset(datasets, separate_eval=False)
        ds.evaluate(fake_inputs)

    with pytest.raises(NotImplementedError):

        @DATASETS.register_module()
        class DummyDataset2:

            def __init__(self):
                self.CLASSES = 0
                self.ann_file = 'empty'

            def __len__(self):
                return 1

            def evaluate(self, res, logger, **kwargs):
                return dict(n=res[0])

        UniformConcatDataset(
            [dict(type='DummyDataset'),
             dict(type='DummyDataset2')],
            show_mean_scores=True)

import os.path as osp
import tempfile

import numpy as np
import pytest

from mmocr.datasets.base_dataset import BaseDataset


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


def test_custom_dataset():
    tmp_dir = tempfile.TemporaryDirectory()
    # create dummy data
    ann_file = osp.join(tmp_dir.name, 'fake_data.txt')
    _create_dummy_ann_file(ann_file)
    loader = _create_dummy_loader()

    for mode in [True, False]:
        dataset = BaseDataset(ann_file, loader, pipeline=[], test_mode=mode)

        # test len
        assert len(dataset) == len(dataset.data_infos)

        # test set group flag
        assert np.allclose(dataset.flag, [0, 0])

        # test prepare_train_img
        expect_results = {
            'img_info': {
                'file_name': 'sample1.jpg',
                'text': 'hello'
            },
            'img_prefix': ''
        }
        assert dataset.prepare_train_img(0) == expect_results

        # test prepare_test_img
        assert dataset.prepare_test_img(0) == expect_results

        # test __getitem__
        assert dataset[0] == expect_results

        # test get_next_index
        assert dataset._get_next_index(0) == 1

        # test format_resuls
        expect_results_copy = {
            key: value
            for key, value in expect_results.items()
        }
        dataset.format_results(expect_results)
        assert expect_results_copy == expect_results

        # test evaluate
        with pytest.raises(NotImplementedError):
            dataset.evaluate(expect_results)

    tmp_dir.cleanup()

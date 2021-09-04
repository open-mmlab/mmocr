# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
import tempfile

import pytest

from mmocr.datasets.utils.loader import HardDiskLoader, LmdbLoader, Loader
from mmocr.utils import lmdb_converter


def _create_dummy_line_str_file(ann_file):
    ann_info1 = 'sample1.jpg hello'
    ann_info2 = 'sample2.jpg world'

    with open(ann_file, 'w') as fw:
        for ann_info in [ann_info1, ann_info2]:
            fw.write(ann_info + '\n')


def _create_dummy_line_json_file(ann_file):
    ann_info1 = {'filename': 'sample1.jpg', 'text': 'hello'}
    ann_info2 = {'filename': 'sample2.jpg', 'text': 'world'}

    with open(ann_file, 'w') as fw:
        for ann_info in [ann_info1, ann_info2]:
            fw.write(json.dumps(ann_info) + '\n')


def test_loader():
    tmp_dir = tempfile.TemporaryDirectory()
    # create dummy data
    ann_file = osp.join(tmp_dir.name, 'fake_data.txt')
    _create_dummy_line_str_file(ann_file)

    parser = dict(
        type='LineStrParser',
        keys=['filename', 'text'],
        keys_idx=[0, 1],
        separator=' ')

    with pytest.raises(AssertionError):
        Loader(ann_file, parser, repeat=0)
    with pytest.raises(AssertionError):
        Loader(ann_file, [], repeat=1)
    with pytest.raises(AssertionError):
        Loader('sample.txt', parser, repeat=1)
    with pytest.raises(NotImplementedError):
        loader = Loader(ann_file, parser, repeat=1)
        print(loader)

    # test text loader and line str parser
    text_loader = HardDiskLoader(ann_file, parser, repeat=1)
    assert len(text_loader) == 2
    assert text_loader.ori_data_infos[0] == 'sample1.jpg hello'
    assert text_loader[0] == {'filename': 'sample1.jpg', 'text': 'hello'}

    # test text loader and linedict parser
    _create_dummy_line_json_file(ann_file)
    json_parser = dict(type='LineJsonParser', keys=['filename', 'text'])
    text_loader = HardDiskLoader(ann_file, json_parser, repeat=1)
    assert text_loader[0] == {'filename': 'sample1.jpg', 'text': 'hello'}

    # test text loader and linedict parser
    _create_dummy_line_json_file(ann_file)
    json_parser = dict(type='LineJsonParser', keys=['filename', 'text'])
    text_loader = HardDiskLoader(ann_file, json_parser, repeat=1)
    it = iter(text_loader)
    with pytest.raises(StopIteration):
        for _ in range(len(text_loader) + 1):
            next(it)

    # test lmdb loader and line str parser
    _create_dummy_line_str_file(ann_file)
    lmdb_file = osp.join(tmp_dir.name, 'fake_data.lmdb')
    lmdb_converter(ann_file, lmdb_file)

    lmdb_loader = LmdbLoader(lmdb_file, parser, repeat=1)
    assert lmdb_loader[0] == {'filename': 'sample1.jpg', 'text': 'hello'}

    tmp_dir.cleanup()

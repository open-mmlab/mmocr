# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
import tempfile

import pytest

from mmocr.datasets.utils.backend import (HardDiskAnnFileBackend,
                                          HTTPAnnFileBackend,
                                          PetrelAnnFileBackend)
from mmocr.datasets.utils.loader import AnnFileLoader
from mmocr.utils import recog2lmdb


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
        AnnFileLoader(ann_file, parser, repeat=0)
    with pytest.raises(AssertionError):
        AnnFileLoader(ann_file, [], repeat=1)

    # test text loader and line str parser
    text_loader = AnnFileLoader(ann_file, parser, repeat=1, file_format='txt')
    assert len(text_loader) == 2
    assert text_loader.ori_data_infos[0] == 'sample1.jpg hello'
    assert text_loader[0] == {'filename': 'sample1.jpg', 'text': 'hello'}

    # test text loader and linedict parser
    _create_dummy_line_json_file(ann_file)
    json_parser = dict(type='LineJsonParser', keys=['filename', 'text'])
    text_loader = AnnFileLoader(
        ann_file, json_parser, repeat=1, file_format='txt')
    assert text_loader[0] == {'filename': 'sample1.jpg', 'text': 'hello'}

    # test text loader and linedict parser
    _create_dummy_line_json_file(ann_file)
    json_parser = dict(type='LineJsonParser', keys=['filename', 'text'])
    text_loader = AnnFileLoader(
        ann_file, json_parser, repeat=1, file_format='txt')
    it = iter(text_loader)
    with pytest.raises(StopIteration):
        for _ in range(len(text_loader) + 1):
            next(it)

    # test lmdb loader and line str parser
    _create_dummy_line_str_file(ann_file)
    lmdb_file = osp.join(tmp_dir.name, 'fake_data.lmdb')
    recog2lmdb(
        img_root=None,
        label_path=ann_file,
        label_format='txt',
        label_only=True,
        output=lmdb_file,
        lmdb_map_size=102400)

    lmdb_loader = AnnFileLoader(
        lmdb_file, parser, repeat=1, file_format='lmdb')
    assert lmdb_loader[0] == {'filename': 'sample1.jpg', 'text': 'hello'}
    lmdb_loader.close()

    with pytest.raises(AssertionError):
        HardDiskAnnFileBackend(file_format='json')
    with pytest.raises(AssertionError):
        PetrelAnnFileBackend(file_format='json')
    with pytest.raises(AssertionError):
        HTTPAnnFileBackend(file_format='json')

    tmp_dir.cleanup()


if __name__ == '__main__':
    test_loader()

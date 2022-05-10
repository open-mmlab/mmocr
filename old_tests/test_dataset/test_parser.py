# Copyright (c) OpenMMLab. All rights reserved.
import json

import pytest

from mmocr.datasets.utils.parser import LineJsonParser, LineStrParser


def test_line_str_parser():
    data_ret = ['sample1.jpg hello\n', 'sample2.jpg world']
    keys = ['filename', 'text']
    keys_idx = [0, 1]
    separator = ' '

    # test init
    with pytest.raises(AssertionError):
        parser = LineStrParser('filename', keys_idx, separator)
    with pytest.raises(AssertionError):
        parser = LineStrParser(keys, keys_idx, [' '])
    with pytest.raises(AssertionError):
        parser = LineStrParser(keys, [0], separator)

    # test get_item
    parser = LineStrParser(keys, keys_idx, separator)
    assert parser.get_item(data_ret, 0) == {
        'filename': 'sample1.jpg',
        'text': 'hello'
    }

    with pytest.raises(Exception):
        parser = LineStrParser(['filename', 'text', 'ignore'], [0, 1, 2],
                               separator)
        parser.get_item(data_ret, 0)


def test_line_dict_parser():
    data_ret = [
        json.dumps({
            'filename': 'sample1.jpg',
            'text': 'hello'
        }),
        json.dumps({
            'filename': 'sample2.jpg',
            'text': 'world'
        })
    ]
    keys = ['filename', 'text']

    # test init
    with pytest.raises(AssertionError):
        parser = LineJsonParser('filename')
    with pytest.raises(AssertionError):
        parser = LineJsonParser([])

    # test get_item
    parser = LineJsonParser(keys)
    assert parser.get_item(data_ret, 0) == {
        'filename': 'sample1.jpg',
        'text': 'hello'
    }

    with pytest.raises(Exception):
        parser = LineJsonParser(['img_name', 'text'])
        parser.get_item(data_ret, 0)

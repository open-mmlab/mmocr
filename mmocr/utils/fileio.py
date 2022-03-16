# Copyright (c) OpenMMLab. All rights reserved.
import json
import os

import mmcv


def list_to_file(filename, lines, file_type='txt'):
    """Write a list of strings to a text file.

    Args:
        filename (str): The output filename. It will be created/overwritten.
        lines (list(str)): Data to be written.
        file_type (str) : Using jsonl or txt to save annotations
    """
    assert filename.split('.')[-1] == file_type
    mmcv.mkdir_or_exist(os.path.dirname(filename))
    with open(filename, 'w', encoding='utf-8') as fw:
        for line in lines:
            if file_type == 'jsonl':
                line = json.dumps(line)
            fw.write(f'{line}\n')


def list_from_file(filename, encoding='utf-8'):
    """Load a text file and parse the content as a list of strings. The
    trailing "\\r" and "\\n" of each line will be removed.

    Note:
        This will be replaced by mmcv's version after it supports encoding.

    Args:
        filename (str): Filename.
        encoding (str): Encoding used to open the file. Default utf-8.

    Returns:
        list[str]: A list of strings.
    """
    item_list = []
    with open(filename, 'r', encoding=encoding) as f:
        for line in f:
            item_list.append(line.rstrip('\n\r'))
    return item_list

# Copyright (c) OpenMMLab. All rights reserved.
import hashlib
import os.path as osp
import sys
from glob import glob
from logging import warning
from typing import List

from mmengine import mkdir_or_exist


def list_to_file(filename, lines):
    """Write a list of strings to a text file.

    Args:
        filename (str): The output filename. It will be created/overwritten.
        lines (list(str)): Data to be written.
    """
    mkdir_or_exist(osp.dirname(filename))
    with open(filename, 'w', encoding='utf-8') as fw:
        for line in lines:
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
    with open(filename, encoding=encoding) as f:
        for line in f:
            item_list.append(line.rstrip('\n\r'))
    return item_list


def is_archive(file_path: str) -> bool:
    """Check whether the file is a supported archive format.

    Args:
        file_path (str): Path to the file.

    Returns:
        bool: Whether the file is an archive.
    """

    suffixes = ['zip', 'tar', 'tar.gz']

    for suffix in suffixes:
        if file_path.endswith(suffix):
            return True
    return False


def check_integrity(file_path: str,
                    md5: str,
                    chunk_size: int = 1024 * 1024) -> bool:
    """Check if the file exist and match to the given md5 code.

    Args:
        file_path (str): Path to the file.
        md5 (str): MD5 to be matched.
        chunk_size (int, optional): Chunk size. Defaults to 1024*1024.

    Returns:
        bool: Whether the md5 is matched.
    """
    if md5 is None:
        warning.warn('MD5 is None, skip the integrity check.')
        return True
    if not osp.exists(file_path):
        return False

    if sys.version_info >= (3, 9):
        hash = hashlib.md5(usedforsecurity=False)
    else:
        hash = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            hash.update(chunk)

    return hash.hexdigest() == md5


def list_files(path: str, suffixes: List) -> List:
    """Retrieve file list from the path.

    Args:
        path (str): Path to the directory.
        suffixes (list[str], optional): Suffixes to be retrieved.

    Returns:
        List: List of the files.
    """

    file_list = []
    for suffix in suffixes:
        file_list.extend(glob(osp.join(path, '*' + suffix)))

    return file_list

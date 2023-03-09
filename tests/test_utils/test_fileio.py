# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import tempfile
import unittest

from mmocr.utils import (check_integrity, get_md5, is_archive, list_files,
                         list_from_file, list_to_file)

lists = [
    [],
    [' '],
    ['\t'],
    ['a'],
    [1],
    [1.],
    ['a', 'b'],
    ['a', 1, 1.],
    [1, 1., 'a'],
    ['啊', '啊啊'],
    ['選択', 'noël', 'Информацией', 'ÄÆä'],
]

dicts = [
    [{
        'text': []
    }],
    [{
        'text': [' ']
    }],
    [{
        'text': ['\t']
    }],
    [{
        'text': ['a']
    }],
    [{
        'text': [1]
    }],
    [{
        'text': [1.]
    }],
    [{
        'text': ['a', 'b']
    }],
    [{
        'text': ['a', 1, 1.]
    }],
    [{
        'text': [1, 1., 'a']
    }],
    [{
        'text': ['啊', '啊啊']
    }],
    [{
        'text': ['選択', 'noël', 'Информацией', 'ÄÆä']
    }],
]


def test_list_to_file():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # test txt
        for i, lines in enumerate(lists):
            filename = f'{tmpdirname}/{i}.txt'
            list_to_file(filename, lines)
            lines2 = [
                line.rstrip('\r\n')
                for line in open(filename, encoding='utf-8').readlines()
            ]
            lines = list(map(str, lines))
            assert len(lines) == len(lines2)
            assert all(line1 == line2 for line1, line2 in zip(lines, lines2))
        # test jsonl
        for i, lines in enumerate(dicts):
            filename = f'{tmpdirname}/{i}.jsonl'
            list_to_file(filename, [json.dumps(line) for line in lines])
            lines2 = [
                json.loads(line.rstrip('\r\n'))['text']
                for line in open(filename, encoding='utf-8').readlines()
            ][0]

            lines = list(lines[0]['text'])
            assert len(lines) == len(lines2)
            assert all(line1 == line2 for line1, line2 in zip(lines, lines2))


def test_list_from_file():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # test txt file
        for i, lines in enumerate(lists):
            filename = f'{tmpdirname}/{i}.txt'
            with open(filename, 'w', encoding='utf-8') as f:
                f.writelines(f'{line}\n' for line in lines)
            lines2 = list_from_file(filename, encoding='utf-8')
            lines = list(map(str, lines))
            assert len(lines) == len(lines2)
            assert all(line1 == line2 for line1, line2 in zip(lines, lines2))
        # test jsonl file
        for i, lines in enumerate(dicts):
            filename = f'{tmpdirname}/{i}.jsonl'
            with open(filename, 'w', encoding='utf-8') as f:
                f.writelines(f'{line}\n' for line in lines)
            lines2 = list_from_file(filename, encoding='utf-8')
            lines = list(map(str, lines))
            assert len(lines) == len(lines2)
            assert all(line1 == line2 for line1, line2 in zip(lines, lines2))


class TestIsArchive(unittest.TestCase):

    def setUp(self) -> None:
        self.zip = 'data/annotations_123.zip'
        self.tar = 'data/img.abc.tar'
        self.targz = 'data/img12345_.tar.gz'
        self.rar = '/m/abc/t.rar'
        self.dir = '/a/b/c/'

    def test_is_archive(self):
        # test zip
        self.assertTrue(is_archive(self.zip))
        # test tar
        self.assertTrue(is_archive(self.tar))
        # test tar.gz
        self.assertTrue(is_archive(self.targz))
        # test rar
        self.assertFalse(is_archive(self.rar))
        # test dir
        self.assertFalse(is_archive(self.dir))


class TestCheckIntegrity(unittest.TestCase):

    def setUp(self) -> None:
        self.file1 = ('tests/data/det_toy_dataset/textdet_test.json',
                      '4185942a3499601e808b26dcca98a4f3')
        self.file2 = ('tests/data/det_toy_dataset/imgs/test/img_1.jpg',
                      'abc123')
        self.file3 = ('abc/abc.jpg', 'abc123')

    def test_check_integrity(self):
        file, md5 = self.file1
        self.assertTrue(check_integrity(file, md5))
        file, md5 = self.file2
        self.assertFalse(check_integrity(file, md5))
        self.assertTrue(check_integrity(file, None))
        file, md5 = self.file3
        self.assertFalse(check_integrity(file, md5))


class TextGetMD5(unittest.TestCase):

    def setUp(self) -> None:
        self.file1 = ('tests/data/det_toy_dataset/textdet_test.json',
                      '4185942a3499601e808b26dcca98a4f3')
        self.file2 = ('tests/data/det_toy_dataset/imgs/test/img_1.jpg',
                      'abc123')

    def test_get_md5(self):
        file, md5 = self.file1
        self.assertEqual(get_md5(file), md5)
        file, md5 = self.file2
        self.assertNotEqual(get_md5(file), md5)


class TestListFiles(unittest.TestCase):

    def setUp(self) -> None:
        self.path = 'tests/data/det_toy_dataset/imgs/test'

    def test_check_integrity(self):
        suffix = 'jpg'
        files = list_files(self.path, suffix)
        for file in os.listdir(self.path):
            if file.endswith(suffix):
                self.assertIn(os.path.join(self.path, file), files)

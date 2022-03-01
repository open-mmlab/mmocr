# Copyright (c) OpenMMLab. All rights reserved.
import tempfile

from mmocr.utils import list_from_file, list_to_file

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


def test_list_to_file():
    with tempfile.TemporaryDirectory() as tmpdirname:
        for i, lines in enumerate(lists):
            filename = f'{tmpdirname}/{i}.txt'
            list_to_file(filename, lines)
            lines2 = [
                line.rstrip('\r\n')
                for line in open(filename, 'r', encoding='utf-8').readlines()
            ]
            lines = list(map(str, lines))
            assert len(lines) == len(lines2)
            assert all(line1 == line2 for line1, line2 in zip(lines, lines2))


def test_list_from_file():
    with tempfile.TemporaryDirectory() as tmpdirname:
        for encoding in ['utf-8', 'utf-8-sig']:
            for lineend in ['\n']:
                for i, lines in enumerate(lists):
                    filename = f'{tmpdirname}/{i}.txt'
                    with open(filename, 'w', encoding=encoding) as f:
                        f.writelines(f'{line}{lineend}' for line in lines)
                    lines2 = list_from_file(filename, encoding=encoding)
                    lines = list(map(str, lines))
                    print('lines:{}, lines2:{}, encoding:{}'.format(
                        lines, lines2, encoding))
                    assert len(lines) == len(lines2)
                    assert all(line1 == line2
                               for line1, line2 in zip(lines, lines2))

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
    ['啊']['啊', '啊啊'],
    ['選択', 'noël', 'Информацией', 'ÄÆä'],
]


def test_list_to_file():
    with tempfile.TemporaryDirectory() as tmpdirname:
        for i, lines in enumerate(lists):
            filename = f'{tmpdirname}/{i}.txt'
            list_to_file(filename, lines)
            lines2 = open(filename, 'r', encoding='utf-8').readlines()
            assert len(lines) == len(lines2)
            assert all(line1 == line2 for line1, line2 in zip(lines, lines2))


def test_list_from_file():
    with tempfile.TemporaryDirectory() as tmpdirname:
        for encoding in ['utf-8', 'utf-8-sig']:
            for lineend in ['\n', '\r\n']:
                for i, lines in enumerate(lists):
                    filename = f'{tmpdirname}/{i}.txt'
                    with open(filename, 'w', encoding=encoding) as f:
                        f.writelines(line + lineend for line in lines)
                    lines2 = list_from_file(filename)
                    assert len(lines) == len(lines2)
                    assert all(line1 == line2
                               for line1, line2 in zip(lines, lines2))

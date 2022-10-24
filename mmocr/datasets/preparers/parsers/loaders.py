# Copyright (c) OpenMMLab. All rights reserved.
import re
from typing import Dict, Tuple, Union

import yaml


def txt_loader(file_path: str,
               separator: str = ',',
               format: str = 'x1,y1,x2,y2,x3,y3,x4,y4,trans',
               encoding='utf-8') -> Union[Dict, str]:
    """Loading txt format annotations.

    Args:
        file_path (str): Path to the txt file.
        separator (str, optional): Separator of data. Defaults to ','.
        format (str, optional): Annotation format.
            Defaults to 'x1,y1,x2,y2,x3,y3,x4,y4,trans'.
        encoding (str, optional): Encoding format. Defaults to 'utf-8'.

    Yields:
        Iterator[Union[Dict, str]]: Original text line or a dict containing
        the information of the text line.
    """
    keys = format.split(separator)
    with open(file_path, 'r', encoding=encoding) as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                yield dict(zip(keys, line.split(separator)))


def totaltext_loader(file_path: str) -> str:
    """The annotation of the totaltext dataset may be stored in multiple lines,
    this loader is designed for this special case.

    Args:
        file_path (str): Path to the txt file

    Yield:
        str: Complete annotation of the txt file
    """

    def parsing_line(line: str) -> Tuple:
        """Parsing a line of the annotation.

        Args:
            line (str): A line of the annotation.

        Returns:
            Tuple: A tuple of (polygon, transcription).
        """
        line = '{' + line.replace('[[', '[').replace(']]', ']') + '}'
        ann_dict = re.sub('([0-9]) +([0-9])', r'\1,\2', line)
        ann_dict = re.sub('([0-9]) +([ 0-9])', r'\1,\2', ann_dict)
        ann_dict = re.sub('([0-9]) -([0-9])', r'\1,-\2', ann_dict)
        ann_dict = ann_dict.replace("[u',']", "[u'#']")
        ann_dict = yaml.safe_load(ann_dict)

        # polygon
        xs, ys = ann_dict['x'], ann_dict['y']
        poly = []
        for x, y in zip(xs, ys):
            poly.append(x)
            poly.append(y)
        # text
        text = ann_dict['transcriptions']
        if len(text) == 0:
            text = '#'
        else:
            word = text[0]
            if len(text) > 1:
                for ann_word in text[1:]:
                    word += ',' + ann_word
            text = str(eval(word))

        return poly, text

    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if idx == 0:
                tmp_line = line
                continue
            if not line.startswith('x:'):
                tmp_line += ' ' + line
                continue
            complete_line = tmp_line
            tmp_line = line
            yield parsing_line(complete_line)

        if tmp_line != '':
            yield parsing_line(tmp_line)

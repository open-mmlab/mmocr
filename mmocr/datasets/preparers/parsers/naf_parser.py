# Copyright (c) OpenMMLab. All rights reserved.
import json
from typing import List, Tuple

import numpy as np

from ..data_preparer import DATA_PARSERS
from .base import BaseParser


@DATA_PARSERS.register_module()
class NAFAnnParser(BaseParser):
    """NAF dataset parser.

    The original annotation format of this dataset is stored in json files,
    which has the following keys that will be used here:
        - 'textBBs': List of text bounding box objects
            - 'poly_points': list of [x,y] pairs, the box corners going
                top-left,top-right,bottom-right,bottom-left
            - 'id': id of the textBB, used to match with the text
        - 'transcriptions': Dict of transcription objects, use the 'id' key
            to match with the textBB.

    Some special characters are used in the transcription:
    "«text»" indicates that "text" had a strikethrough
    "¿" indicates the transcriber could not read a character
    "§" indicates the whole line or word was illegible
    "" (empty string) is if the field was blank

    Args:
        ignore (list(str)): The text of the ignored instances. Default: ['#'].
        det (bool): Whether to parse the detection annotation. Default: True.
            If False, the parser will consider special case in NAF dataset
            where the transcription is not available.
    """

    def __init__(self,
                 ignore: List[str] = ['#'],
                 det: bool = True,
                 **kwargs) -> None:
        self.ignore = ignore
        self.det = det
        super().__init__(**kwargs)

    def parse_file(self, img_path: str, ann_path: str) -> Tuple:
        """Convert single annotation."""
        instances = list()
        for poly, text in self.loader(ann_path):
            instances.append(
                dict(poly=poly, text=text, ignore=text in self.ignore))

        return img_path, instances

    def loader(self, file_path: str) -> str:
        """Load the annotation of the NAF dataset.

        Args:
            file_path (str): Path to the json file

        Retyrb:
            str: Complete annotation of the json file
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        # 'textBBs' contains the printed texts of the table while 'fieldBBs'
        #  contains the text filled by human.
        for box_type in ['textBBs', 'fieldBBs']:
            if not self.det:
                # 'textBBs' is only used for detection task.
                if box_type == 'textBBs':
                    continue
            for anno in data[box_type]:
                # Skip blanks
                if self.det:
                    if box_type == 'fieldBBs':
                        if anno['type'] == 'blank':
                            continue
                    poly = np.array(anno['poly_points']).reshape(
                        1, 8)[0].tolist()
                    # Since detection task only need poly, we can skip the
                    # transcription part that can be empty.
                    text = None
                else:
                    # For tasks that need transcription, NAF dataset has
                    # serval special cases:
                    # 1. The transcription for the whole image is not
                    # available.
                    # 2. The transcription for the certain text is not
                    # available.
                    # 3. If the length of the transcription is 0, it should
                    # be ignored.
                    if 'transcriptions' not in data.keys():
                        break
                    if anno['id'] not in data['transcriptions'].keys():
                        continue
                    text = data['transcriptions'][anno['id']]
                    text = text.strip(
                        '\u202a')  # Remove unicode control character
                    text = text.replace('»', '').replace(
                        '«', '')  # Remove strikethrough flag
                    if len(text) == 0:
                        continue
                    poly = np.array(anno['poly_points']).reshape(
                        1, 8)[0].tolist()
                yield poly, text

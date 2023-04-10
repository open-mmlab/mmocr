# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List

from mmocr.registry import DATA_PARSERS
from .icdar_txt_parser import ICDARTxtTextRecogAnnParser


@DATA_PARSERS.register_module()
class MJSynthAnnParser(ICDARTxtTextRecogAnnParser):
    """MJSynth Text Recognition Annotation Parser.

    The original annotation format of this dataset is stored in txt files,
    which is formed as the following format:
        img_path, transcription

    Args:
        separator (str): The separator between each element in a line. Defaults
            to ','.
        ignore (str): The text to be ignored. Defaults to '#'.
        format (str): The format of the annotation. Defaults to 'img, text'.
        encoding (str): The encoding of the annotation file. Defaults to
            'utf-8-sig'.
        nproc (int): The number of processes to parse the annotation. Defaults
            to 1.
        base_name (bool): Whether to use the basename of the image path as the
            image name. Defaults to False.
        remove_strs (List[str], Optional): Used to remove redundant strings in
            the transcription. Defaults to ['"'].
    """

    def parse_files(self, img_dir: str, ann_path: str) -> List:
        """Parse annotations."""
        assert isinstance(ann_path, str)
        samples = list()
        for anno in self.loader(
                file_path=ann_path,
                format=self.format,
                encoding=self.encoding,
                separator=self.sep):
            text = osp.basename(anno['img']).split('_')[1]
            if self.remove_strs is not None:
                for strs in self.remove_strs:
                    text = text.replace(strs, '')
            if text == self.ignore:
                continue
            img_name = anno['img']
            samples.append((osp.join(img_dir, img_name), text))

        return samples

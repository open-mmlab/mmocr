# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import numpy as np
from mmengine import track_parallel_progress
from scipy.io import loadmat

from mmocr.utils import is_type_list
from ..data_preparer import DATA_PARSERS
from .base import BaseParser


@DATA_PARSERS.register_module()
class SynthTextTextDetAnnParser(BaseParser):
    """SynthText Text Detection Annotation Parser.

    Args:
        separator (str): The separator between each element in a line. Defaults
            to ','.
        ignore (str): The text to be ignored. Defaults to '###'.
        format (str): The format of the annotation. Defaults to
            'x1,y1,x2,y2,x3,y3,x4,trans'.
        encoding (str): The encoding of the annotation file. Defaults to
            'utf-8-sig'.
        nproc (int): The number of processes to parse the annotation. Defaults
            to 1.
        remove_strs (List[str], Optional): Used to remove redundant strings in
            the transcription. Defaults to None.
        mode (str, optional): The mode of the box converter. Supported modes
            are 'xywh' and 'xyxy'. Defaults to None.
    """

    def __init__(self,
                 separator: str = ',',
                 ignore: str = '###',
                 format: str = 'x1,y1,x2,y2,x3,y3,x4,y4,trans',
                 encoding: str = 'utf-8',
                 nproc: int = 1,
                 remove_strs: Optional[List[str]] = None,
                 mode: str = None) -> None:
        self.sep = separator
        self.format = format
        self.encoding = encoding
        self.ignore = ignore
        self.mode = mode
        self.remove_strs = remove_strs
        super().__init__(nproc=nproc)

    def _trace_boundary(self, char_boxes: List[np.ndarray]) -> np.ndarray:
        """Trace the boundary point of text.

        Args:
            char_boxes (list[ndarray]): The char boxes for one text. Each
                element is 4x2 ndarray.

        Returns:
            ndarray: The boundary point sets with size nx2.
        """
        assert is_type_list(char_boxes, np.ndarray)

        # from top left to to right
        p_top = [box[0:2] for box in char_boxes]
        # from bottom right to bottom left
        p_bottom = [
            char_boxes[idx][[2, 3], :]
            for idx in range(len(char_boxes) - 1, -1, -1)
        ]

        p = p_top + p_bottom

        boundary = np.concatenate(p).astype(int)

        return boundary

    def _match_bbox_char_str(self, bboxes: np.ndarray, char_bboxes: np.ndarray,
                             strs: np.ndarray
                             ) -> Tuple[List[np.ndarray], List[str]]:
        """Match the bboxes, char bboxes, and strs.

        Args:
            bboxes (ndarray): The text boxes of size (2, 4, num_box).
            char_bboxes (ndarray): The char boxes of size (2, 4, num_char_box).
            strs (ndarray): The string of size (num_strs,)

        Returns:
            Tuple(List[ndarray], List[str]): Polygon & word list.
        """
        assert isinstance(bboxes, np.ndarray)
        assert isinstance(char_bboxes, np.ndarray)
        assert isinstance(strs, np.ndarray)
        # bboxes = bboxes.astype(np.int32)
        char_bboxes = char_bboxes.astype(np.int32)

        if len(char_bboxes.shape) == 2:
            char_bboxes = np.expand_dims(char_bboxes, axis=2)
        char_bboxes = np.transpose(char_bboxes, (2, 1, 0))
        num_boxes = 1 if len(bboxes.shape) == 2 else bboxes.shape[-1]

        poly_charbox_list = [[] for _ in range(num_boxes)]

        words = []
        for line in strs:
            words += line.split()
        words_len = [len(w) for w in words]
        words_end_inx = np.cumsum(words_len)
        start_inx = 0
        for word_inx, end_inx in enumerate(words_end_inx):
            for char_inx in range(start_inx, end_inx):
                poly_charbox_list[word_inx].append(char_bboxes[char_inx])
            start_inx = end_inx

        for box_inx in range(num_boxes):
            assert len(poly_charbox_list[box_inx]) > 0

        poly_boundary_list = []
        for item in poly_charbox_list:
            boundary = np.ndarray((0, 2))
            if len(item) > 0:
                boundary = self._trace_boundary(item)
            poly_boundary_list.append(boundary)

        return poly_boundary_list, words

    def parse_files(self, files: List[Tuple], split: str) -> List[Tuple]:
        """Convert annotations to MMOCR format.

        Args:
            files (Tuple): A list of tuple of path to image and annotation.

        Returns:
            List[Tuple]: A list of a tuple of (image_path, instances)
        """
        assert isinstance(files, str)
        gt = loadmat(files)
        samples = track_parallel_progress(
            self.parse_file,
            list(
                zip(gt['imnames'][0], gt['wordBB'][0], gt['charBB'][0],
                    gt['txt'][0])),
            nproc=self.nproc)
        return samples

    def parse_file(self, annotation: Tuple) -> Tuple:
        """Parse single annotation."""
        img_file, wordBB, charBB, txt = annotation
        polys_list, word_list = self._match_bbox_char_str(wordBB, charBB, txt)

        instances = list()
        for poly, word in zip(polys_list, word_list):
            instances.append(
                dict(poly=poly.flatten().tolist(), text=word, ignore=False))
        return img_file[0], instances

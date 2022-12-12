# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

from mmocr.utils.progressbar import track_parallel_progress_multi_args


class BaseParser:
    """Base class for parsing annotations.

    Args:
        data_root (str, optional): Path to the data root. Defaults to None.
        nproc (int, optional): Number of processes. Defaults to 1.
    """

    def __init__(self, nproc: int = 1) -> None:
        self.nproc = nproc

    def __call__(self, files: List[Tuple], split: str) -> List:
        """Parse annotations.

        Args:
            files (List[Tuple]): A list of a tuple of
                (image_path, annotation_path).
            split (str): The split of the dataset.

        Returns:
            List: A list of a tuple of (image_path, instances)
        """
        samples = self.parse_files(files, split)
        return samples

    def parse_files(self,
                    img_paths: Union[List[str], str],
                    ann_paths: Union[List[str], str],
                    split: Optional[str] = None) -> List[Tuple]:
        """Convert annotations to MMOCR format.

        Args:
            files (Tuple): A list of tuple of path to image and annotation.

        Returns:
            List[Tuple]: A list of a tuple of (image_path, instances)
        """
        func = partial(self.parse_file, split=split)
        files = list(zip(img_paths, ann_paths))
        samples = track_parallel_progress_multi_args(
            func, files, nproc=self.nproc)
        return samples

    @abstractmethod
    def parse_file(self, img_path: str, ann_path: str, split: str) -> Tuple:
        """Convert annotation for a single image.

        Args:
            file (Tuple): A tuple of path to image and annotation
            split (str): Current split.

        Returns:
            Tuple: A tuple of (img_path, instance). Instance is a list of dict
            containing parsed annotations, which should contain the
            following keys:
            - 'poly' or 'box' (textdet or textspotting)
            - 'text' (textspotting or textrecog)
            - 'ignore' (all task)

        Examples:
        An example of returned values:
        >>> ('imgs/train/xxx.jpg',
        >>> dict(
        >>>    poly=[[[0, 1], [1, 1], [1, 0], [0, 0]]],
        >>>    text='hello',
        >>>    ignore=False)
        >>> )
        """
        raise NotImplementedError

    def loader(self,
               file_path: str,
               separator: str = ',',
               format: str = 'x1,y1,x2,y2,x3,y3,x4,y4,trans',
               encoding='utf-8') -> Union[Dict, str]:
        """A basic loader designed for .txt format annotation. It greedily
        extracts information separated by separators.

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
                values = line.split(separator)
                values = values[:len(keys) -
                                1] + [separator.join(values[len(keys) - 1:])]
                if line:
                    yield dict(zip(keys, values))

# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

from mmengine import track_parallel_progress


class BaseParser:
    """Base class for parsing annotations.

    Args:
        data_root (str, optional): Path to the data root. Defaults to None.
        nproc (int, optional): Number of processes. Defaults to 1.
    """

    def __init__(self,
                 data_root: Optional[str] = None,
                 nproc: int = 1) -> None:
        self.data_root = data_root
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

    def parse_files(self, files: List[Tuple], split: str) -> List[Tuple]:
        """Convert annotations to MMOCR format.

        Args:
            files (Tuple): A tuple of path to image and annotation.

        Returns:
            List[Tuple]: A list of a tuple of (image_path, instances)
        """
        func = partial(self.parse_file, split=split)
        samples = track_parallel_progress(func, files, nproc=self.nproc)
        return samples

    @abstractmethod
    def parse_file(self, file: Tuple, split: str) -> Dict:
        """Convert annotation for a single image."""
        raise NotImplementedError

    def loader(self,
               file_path: str,
               separator: str = ',',
               format: str = 'x1,y1,x2,y2,x3,y3,x4,y4,trans',
               encoding='utf-8') -> Union[Dict, str]:
        """A basic loader designed for .txt format annotation.

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

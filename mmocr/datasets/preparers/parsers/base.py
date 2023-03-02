# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import Dict, List, Tuple, Union

from mmocr.utils import track_parallel_progress_multi_args


class BaseParser:
    """Base class for parsing annotations.

    Args:
        split (str): The split of the dataset. It usually be set automatically
            and users do not need to set it manually in config file in most
            cases.
        nproc (int): Number of processes to process the data. Defaults to 1.
            It usually be set automatically and users do not need to set it
            manually in config file in most cases.
    """

    def __init__(self, split: str, nproc: int = 1) -> None:
        self.nproc = nproc
        self.split = split

    def __call__(self, img_paths: Union[List[str], str],
                 ann_paths: Union[List[str], str]) -> List[Tuple]:
        """Parse annotations.

        Args:
            img_paths (str or list[str]): the list of image paths or the
                directory of the images.
            ann_paths (str or list[str]): the list of annotation paths or the
                path of the annotation file which contains all the annotations.

        Returns:
            List: A list of a tuple of (image_path, instances)
        """
        samples = self.parse_files(img_paths, ann_paths)
        return samples

    def parse_files(self, img_paths: Union[List[str], str],
                    ann_paths: Union[List[str], str]) -> List[Tuple]:
        """Convert annotations to MMOCR format.

        Args:
            img_paths (str or list[str]): the list of image paths or the
                directory of the images.
            ann_paths (str or list[str]): the list of annotation paths or the
                path of the annotation file which contains all the annotations.

        Returns:
            List[Tuple]: A list of a tuple of (image_path, instances).

            - img_path (str): The path of image file, which can be read
              directly by opencv.
            - instance: instance is a list of dict containing parsed
              annotations, which should contain the following keys:

              - 'poly' or 'box' (textdet or textspotting)
              - 'text' (textspotting or textrecog)
              - 'ignore' (all task)
        """
        samples = track_parallel_progress_multi_args(
            self.parse_file, (img_paths, ann_paths), nproc=self.nproc)
        return samples

    @abstractmethod
    def parse_file(self, img_path: str, ann_path: str) -> Tuple:
        """Convert annotation for a single image.

        Args:
            img_path (str): The path of image.
            ann_path (str): The path of annotation.

        Returns:
            Tuple: A tuple of (img_path, instance).

            - img_path (str): The path of image file, which can be read
              directly by opencv.
            - instance: instance is a list of dict containing parsed
              annotations, which should contain the following keys:

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

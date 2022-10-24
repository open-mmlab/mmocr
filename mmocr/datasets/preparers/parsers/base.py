# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from functools import partial
from typing import Dict, List, Optional, Tuple

from mmengine import track_parallel_progress


class BaseParser:

    def __init__(self,
                 data_root: Optional[str] = None,
                 nproc: int = 1) -> None:
        self.data_root = data_root
        self.nproc = nproc

    def __call__(self, files: List[Tuple], split: str) -> List:
        samples = self.parse_files(files, split)
        return samples

    def parse_files(self, files: List[Tuple], split: str) -> List[Tuple]:
        """Convert annotations to MMOCR format.

        Args:
            files (Tuple): A tuple of path to image and annotation.

        Returns:
            List[Tuple]: A list of a tuple of (image_path, instance)
        """
        func = partial(self.parse_file, split=split)
        samples = track_parallel_progress(func, files, nproc=self.nproc)
        return samples

    @abstractmethod
    def parse_file(self, file: Tuple, split: str) -> Dict:
        """Convert annotation for a single image."""
        raise NotImplementedError

# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import Dict, List, Tuple

from mmengine import track_parallel_progress


class BasePacker:
    """Base class for packing the parsed annotation info to MMOCR format.

    Args:
        data_root (str): The root path of the dataset.  It usually be set auto-
            matically and users do not need to set it manually in config file
            in most cases.
        split (str): The split of the dataset. It usually be set automatically
            and users do not need to set it manually in config file in most
            cases.
        nproc (int): Number of processes to process the data. Defaults to 1.
            It usually be set automatically and users do not need to set it
            manually in config file in most cases.
    """

    def __init__(self, data_root: str, split: str, nproc: int = 1) -> None:
        self.data_root = data_root
        self.split = split
        self.nproc = nproc

    @abstractmethod
    def pack_instance(self, sample: Tuple, split: str) -> Dict:
        """Pack the parsed annotation info to an MMOCR format instance.

        Args:
            sample (Tuple): A tuple of (img_file, ann_file).
               - img_path (str): Path to image file.
               - instances (Sequence[Dict]): A list of converted annos.
            split (str): The split of the instance.

        Returns:
            Dict: An MMOCR format instance.
        """

    @abstractmethod
    def add_meta(self, sample: List) -> Dict:
        """Add meta information to the sample.

        Args:
            sample (List): A list of samples of the dataset.

        Returns:
            Dict: A dict contains the meta information and samples.
        """

    def __call__(self, samples) -> Dict:
        samples = track_parallel_progress(
            self.pack_instance, samples, nproc=self.nproc)
        samples = self.add_meta(samples)
        return samples

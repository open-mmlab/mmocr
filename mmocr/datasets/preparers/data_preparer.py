# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import time

from mmengine import Registry
from mmengine.config import Config

DATA_OBTAINERS = Registry('data_obtainer')
DATA_CONVERTERS = Registry('data_converter')
DATA_PARSERS = Registry('data_parser')
DATA_DUMPERS = Registry('data_dumper')


class DatasetPreparer:
    """Base class of dataset preparer.

    Dataset preparer is used to prepare dataset for MMOCR. It mainly consists
    of two steps:

      1. Obtain the dataset
            - Download
            - Extract
            - Move/Rename
      2. Process the dataset
            - Parse original annotations
            - Convert to mmocr format
            - Dump the annotation file
            - Clean useless files

    After all these steps, the original datasets have been prepared for
    usage in MMOCR. Check out the dataset format used in MMOCR here:
    https://mmocr.readthedocs.io/en/dev-1.x/user_guides/dataset_prepare.html
    """

    def __init__(self,
                 cfg_path: str,
                 dataset_name: str,
                 task: str = 'textdet',
                 nproc: int = 4) -> None:
        """Initialization. Load necessary meta info and print license.

        Args:
            cfg_path (str): Path to dataset config file.
            dataset_name (str): Dataset name.
            task (str): Task type. Options are 'textdet', 'textrecog',
                'textspotter', and 'kie'. Defaults to 'textdet'.
            nproc (int): Number of parallel processes. Defaults to 4.
        """
        cfg_path = osp.join(cfg_path, dataset_name)
        self.nproc = nproc
        self.task = task
        self.parse_meta(cfg_path)
        self.parse_cfg(cfg_path)

    def __call__(self):
        """Prepare the dataset."""
        if self.with_obtainer:
            self.data_obtainer()
        if self.with_converter:
            self.data_converter()

    def parse_meta(self, cfg_path: str) -> None:
        """Parse meta file.

        Args:
            cfg_path (str): Path to meta file.
        """
        try:
            meta = Config.fromfile(osp.join(cfg_path, 'metafile.yml'))
        except FileNotFoundError:
            return
        assert self.task in meta['Data']['Tasks'], \
            f'Task {self.task} not supported!'
        # License related
        if meta['Data']['License']['Type']:
            print(f"\033[1;33;40mDataset Name: {meta['Name']}")
            print(f"License Type: {meta['Data']['License']['Type']}")
            print(f"License Link: {meta['Data']['License']['Link']}")
            print(f"BibTeX: {meta['Paper']['BibTeX']}\033[0m")
            print(
                '\033[1;31;43mMMOCR does not own the dataset. Using this '
                'dataset you must accept the license provided by the owners, '
                'and cite the corresponding papers appropriately.')
            print('If you do not agree with the above license, please cancel '
                  'the progress immediately by pressing ctrl+c. Otherwise, '
                  'you are deemed to accept the terms and conditions.\033[0m')
            for i in range(5):
                print(f'{5-i}...')
                time.sleep(1)

    def parse_cfg(self, cfg_path: str) -> None:
        """Parse dataset config file.

        Args:
            cfg_path (str): Path to dataset config file.
        """
        cfg = Config.fromfile(osp.join(cfg_path, self.task + '.py'))

        if 'data_obtainer' in cfg:
            self.data_obtainer = DATA_OBTAINERS.build(cfg.data_obtainer)
        if 'data_converter' in cfg:
            cfg.data_converter.update(dict(nproc=self.nproc))
            self.data_converter = DATA_CONVERTERS.build(cfg.data_converter)

    @property
    def with_obtainer(self) -> bool:
        """bool: whether the data preparer has an obtainer"""
        return getattr(self, 'data_obtainer', None) is not None

    @property
    def with_converter(self) -> bool:
        """bool: whether the data preparer has an converter"""
        return getattr(self, 'data_converter', None) is not None

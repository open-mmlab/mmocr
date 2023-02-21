# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import List, Union

from mmengine import Registry

from mmocr.utils.typing_utils import ConfigType, OptConfigType

DATA_PREPARERS = Registry('data preparer')
DATA_OBTAINERS = Registry('data_obtainer')
DATA_GATHERERS = Registry('data_gatherer')
DATA_PARSERS = Registry('data_parser')
DATA_DUMPERS = Registry('data_dumper')
DATA_PACKERS = Registry('data_packer')
CFG_GENERATORS = Registry('cfg_generator')


@DATA_PREPARERS.register_module()
class DatasetPreparer:
    """Base class of dataset preparer.

    Dataset preparer is used to prepare dataset for MMOCR. It mainly consists
    of three step:
      1. For each split:
        - Obtain the dataset
            - Download
            - Extract
            - Move/Rename
        - Gather the dataset
        - Parse the dataset
        - Pack the dataset to MMOCR format
        - Dump the dataset
      2. Clean useless files
      3. Generate the base config for this dataset

    After all these steps, the original datasets have been prepared for
    usage in MMOCR. Check out the dataset format used in MMOCR here:
    https://mmocr.readthedocs.io/en/dev-1.x/user_guides/dataset_prepare.html

    Args:
        data_root (str): Root directory of data.
        dataset_name (str): Dataset name.
        task (str): Task type. Options are 'textdet', 'textrecog',
            'textspotter', and 'kie'. Defaults to 'textdet'.
        nproc (int): Number of parallel processes. Defaults to 4.
        train_preparer(OptConfigType): cfg for train data prepare.
            - obtainer:
            - gatherer
            - parser:
            - packer:
            - dumper:
        test_preparer(OptConfigType): cfg for test data prepare.
            - obtainer:
            - gatherer
            - parser:
            - packer:
            - dumper:
        val_preparer(OptConfigType): cfg for train data prepare.
            - obtainer:
            - gatherer
            - parser:
            - packer:
            - dumper:
    """

    def __init__(self,
                 data_root: str,
                 dataset_name: str = '',
                 task: str = 'textdet',
                 nproc: int = 4,
                 train_preparer: OptConfigType = None,
                 test_preparer: OptConfigType = None,
                 val_preparer: OptConfigType = None,
                 config_generator: OptConfigType = None) -> None:
        self.data_root = data_root
        self.nproc = nproc
        self.task = task
        self.dataset_name = dataset_name
        self.train_preparer = train_preparer
        self.test_preparer = test_preparer
        self.val_preparer = val_preparer
        self.config_generator = config_generator

    def run(self, splits: Union[str, List] = ['train', 'test', 'val']) -> None:
        """Prepare the dataset."""
        if isinstance(splits, str):
            splits = [splits]
        assert set(splits).issubset(set(['train', 'test',
                                         'val'])), 'Invalid split name'
        for split in splits:
            self.loop(split, getattr(self, f'{split}_preparer'))
        self.generate_config()

    @classmethod
    def from_file(cls, cfg: ConfigType) -> 'DatasetPreparer':
        """Create a DataPreparer from config file.

        Args:
            cfg (ConfigType): A config used for building runner. Keys of
                ``cfg`` can see :meth:`__init__`.

        Returns:
            Runner: A DatasetPreparer build from ``cfg``.
        """

        cfg = copy.deepcopy(cfg)
        data_preparer = cls(
            data_root=cfg['data_root'],
            dataset_name=cfg.get('dataset_name', ''),
            task=cfg.get('task', 'textdet'),
            nproc=cfg.get('nproc', 4),
            train_preparer=cfg.get('train_preparer', None),
            test_preparer=cfg.get('test_preparer', None),
            val_preparer=cfg.get('val_preparer', None),
            config_generator=cfg.get('config_generator', None))
        return data_preparer

    def loop(self, split: str, cfg: ConfigType) -> None:
        """Loop over the dataset.

        Args:
            split (str): The split of the dataset.
            cfg (ConfigType): A config used for building obtainer, gatherer,
                parser, packer and dumper.
        """
        if cfg is None:
            return

        # build obtainer and run
        obtainer = cfg.get('obtainer', None)
        if obtainer:
            print(f'Obtaining {split} Dataset...')
            obtainer.setdefault('task', default=self.task)
            obtainer.setdefault('data_root', default=self.data_root)
            obtainer = DATA_OBTAINERS.build(obtainer)
            obtainer()

        # build gatherer
        gatherer = cfg.get('gatherer', None)
        parser = cfg.get('parser', None)
        packer = cfg.get('packer', None)
        dumper = cfg.get('dumper', None)
        related = [gatherer, parser, packer, dumper]
        if not (all(item is None
                    for item in related) or all(item is not None
                                                for item in related)):
            raise ValueError('gatherer, parser, packer and dumper should be '
                             'either all None or not None')

        if all(item is None for item in related):  # no data process
            return

        print(f'Gathering {split} Dataset...')
        gatherer.setdefault('split', default=split)
        gatherer.setdefault('data_root', default=self.data_root)
        gatherer.setdefault('ann_dir', default='annotations')
        gatherer.setdefault(
            'img_dir', default=osp.join(f'{self.task}_imgs', split))

        gatherer = DATA_GATHERERS.build(gatherer)
        img_paths, ann_paths = gatherer()

        # build parser
        print(f'Parsing {split} Images and Annotations...')
        parser.setdefault('split', default=split)
        parser.setdefault('nproc', default=self.nproc)
        parser = DATA_PARSERS.build(parser)
        # Convert dataset annotations to MMOCR format
        samples = parser.parse_files(img_paths, ann_paths)

        # build packer
        print(f'Packing {split} Annotations...')
        packer.setdefault('split', default=split)
        packer.setdefault('nproc', default=self.nproc)
        packer.setdefault('data_root', default=self.data_root)
        packer = DATA_PACKERS.build(packer)
        samples = packer(samples)

        # build dumper
        print(f'Dumping {split} Annotations...')
        # Dump annotation files
        dumper.setdefault('task', default=self.task)
        dumper.setdefault('split', default=split)
        dumper.setdefault('data_root', default=self.data_root)
        dumper = DATA_DUMPERS.build(dumper)
        dumper.dump(samples)

    def generate_config(self):
        if self.config_generator is None:
            return
        self.config_generator.setdefault(
            'dataset_name', default=self.dataset_name)
        self.config_generator.setdefault('data_root', default=self.data_root)
        config_generator = CFG_GENERATORS.build(self.config_generator)
        print('Generating base configs...')
        config_generator()

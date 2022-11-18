# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from abc import abstractmethod
from typing import Dict, List, Optional

from mmengine import mkdir_or_exist

from .data_preparer import CFG_GENERATORS


class BaseConfigGenerator:
    """Base class for config generator.

    Args:
        data_root (str): The root path of the dataset.
        task (str): The task of the dataset.
        dataset_name (str): The name of the dataset.
        train_anns (List[Dict], optional): A list of train annotation files
            to appear in the base configs. Defaults to None.
            Each element is typically a dict with the following fields:
            - file (str): The path to the annotation file relative to
              data_root.
            - prefix (str, optional): Affects the prefix of the resulting
              variable in the generated config. Defaults to be the same as
              ``dataset_name``.
        val_anns (List[Dict], optional): A list of val annotation files
            to appear in the base configs, similar to ``train_anns``. Defaults
            to None.
        test_anns (List[Dict], optional): A list of test annotation files
            to appear in the base configs, similar to ``train_anns``. Defaults
            to None.
        config_path (str): Path to the configs. Defaults to 'configs/'.
    """

    def __init__(
        self,
        data_root: str,
        task: str,
        dataset_name: str,
        train_anns: Optional[List[Dict]] = None,
        val_anns: Optional[List[Dict]] = None,
        test_anns: Optional[List[Dict]] = None,
        config_path: str = 'configs/',
    ) -> None:
        self.config_path = config_path
        self.data_root = data_root
        self.task = task
        self.dataset_name = dataset_name
        self._prepare_anns(train_anns, val_anns, test_anns)

    def _prepare_anns(self, train_anns: Optional[List[Dict]],
                      val_anns: Optional[List[Dict]],
                      test_anns: Optional[List[Dict]]) -> None:
        """Preprocess input arguments and stores these information into
        ``self.anns``.

        ``self.anns`` is a that maps the name of a dataset config variable to
        a dict, which contains the following fields:
        - file (str): The path to the annotation file relative to
          data_root.
        - split (str): The split the annotation belongs to. Usually
          it can be 'train', 'val' and 'test'.
        - prefix (str, optional): Affects the prefix of the resulting
          variable in the generated config. Defaults to be the same as
          ``self.dataset_name``.
        """
        self.anns = {}
        for split, ann_list in zip(('train', 'val', 'test'),
                                   (train_anns, val_anns, test_anns)):
            if ann_list is None:
                continue
            if not isinstance(ann_list, list):
                raise ValueError(f'{split}_anns must be either a list or'
                                 ' None!')
            for ann_dict in ann_list:
                assert 'file' in ann_dict
                if 'prefix' not in ann_dict:
                    ann_dict['prefix'] = self.dataset_name
                ann_dict['split'] = split
                key = f'{ann_dict["prefix"]}_{self.task}_{split}'
                if key in self.anns:
                    raise ValueError(
                        f'Duplicate data prefix {ann_dict["prefix"]} in'
                        f'{split} found!')
                self.anns[key] = ann_dict

    def __call__(self) -> None:
        """Generates the base dataset config."""

        dataset_config = self._gen_dataset_config()

        cfg_path = osp.join(self.config_path, self.task, '_base_', 'datasets',
                            f'{self.dataset_name}.py')
        if osp.exists(cfg_path):
            while True:
                c = input(f'{cfg_path} already exists, overwrite? (Y/n) ') \
                    or 'Y'
                if c.lower() == 'y':
                    break
                if c.lower() == 'n':
                    return
        mkdir_or_exist(osp.dirname(cfg_path))
        with open(cfg_path, 'w') as f:
            f.write(
                f'{self.dataset_name}_{self.task}_data_root = \'{self.data_root}\'\n'  # noqa: E501
            )
            f.write(dataset_config)

    @abstractmethod
    def _gen_dataset_config(self) -> str:
        """Generate a full dataset config based on the annotation file
        dictionary.

        Returns:
            str: The generated dataset config.
        """


@CFG_GENERATORS.register_module()
class TextDetConfigGenerator(BaseConfigGenerator):
    """Text detection config generator.

    Args:
        data_root (str): The root path of the dataset.
        dataset_name (str): The name of the dataset.
        train_anns (List[Dict], optional): A list of train annotation files
            to appear in the base configs. Defaults to
            ``[dict(file='textdet_train.json')]``.
            Each element is typically a dict with the following fields:
            - file (str): The path to the annotation file relative to
              data_root.
            - prefix (str, optional): Affects the prefix of the resulting
              variable in the generated config. Defaults to be the same as
              ``dataset_name``.
        val_anns (List[Dict], optional): A list of val annotation files
            to appear in the base configs, similar to ``train_anns``. Defaults
            to [].
        test_anns (List[Dict], optional): A list of test annotation files
            to appear in the base configs, similar to ``train_anns``. Defaults
            to ``[dict(file='textdet_test.json')]``.
        config_path (str): Path to the configs. Defaults to 'configs/'.
    """

    def __init__(
        self,
        data_root: str,
        dataset_name: str,
        train_anns: Optional[List[Dict]] = [dict(file='textdet_train.json')],
        val_anns: Optional[List[Dict]] = [],
        test_anns: Optional[List[Dict]] = [dict(file='textdet_test.json')],
        config_path: str = 'configs/',
    ) -> None:
        super().__init__(
            data_root=data_root,
            task='textdet',
            dataset_name=dataset_name,
            train_anns=train_anns,
            val_anns=val_anns,
            test_anns=test_anns,
            config_path=config_path,
        )

    def _gen_dataset_config(self) -> str:
        """Generate a full dataset config based on the annotation file
        dictionary.

        Args:
            ann_dict (dict[str, dict(str, str)]): A nested dictionary that maps
                a config variable name (such as icdar2015_textrecog_train) to
                its corresponding annotation information dict. Each dict
                contains following keys:
                - file (str): The path to the annotation file relative to
                data_root.
                - split (str): The split the annotation belongs to. Usually
                it can be 'train', 'val' and 'test'.
                - prefix (str): Affects the prefix of the resulting
                variable in the generated config.

        Returns:
            str: The generated dataset config.
        """
        cfg = ''
        for key_name, ann_dict in self.anns.items():
            cfg += f'\n{key_name} = dict(\n'
            cfg += '    type=\'OCRDataset\',\n'
            cfg += '    data_root=' + f'{self.dataset_name}_{self.task}_data_root,\n'  # noqa: E501
            cfg += f'    ann_file=\'{ann_dict["file"]}\',\n'
            if ann_dict['split'] == 'train':
                cfg += '    filter_cfg=dict(filter_empty_gt=True, min_size=32),\n'  # noqa: E501
            elif ann_dict['split'] in ['test', 'val']:
                cfg += '    test_mode=True,\n'
            cfg += '    pipeline=None)\n'
        return cfg


@CFG_GENERATORS.register_module()
class TextRecogConfigGenerator(BaseConfigGenerator):
    """Text recognition config generator.

    Args:
        data_root (str): The root path of the dataset.
        dataset_name (str): The name of the dataset.
        train_anns (List[Dict], optional): A list of train annotation files
            to appear in the base configs. Defaults to
            ``[dict(file='textrecog_train.json')]``.
            Each element is typically a dict with the following fields:
            - file (str): The path to the annotation file relative to
              data_root.
            - prefix (str, optional): Affects the prefix of the resulting
              variable in the generated config. Defaults to be the same as
              ``dataset_name``.
        val_anns (List[Dict], optional): A list of val annotation files
            to appear in the base configs, similar to ``train_anns``. Defaults
            to [].
        test_anns (List[Dict], optional): A list of test annotation files
            to appear in the base configs, similar to ``train_anns``. Defaults
            to ``[dict(file='textrecog_test.json')]``.
        config_path (str): Path to the configs. Defaults to 'configs/'.

    Example:
        It generates a dataset config like:
        >>> ic15_rec_data_root = 'data/icdar2015/'
        >>> icdar2015_textrecog_train = dict(
        >>>     type='OCRDataset',
        >>>     data_root=ic15_rec_data_root,
        >>>     ann_file='textrecog_train.json',
        >>>     test_mode=False,
        >>>     pipeline=None)
        >>> icdar2015_textrecog_test = dict(
        >>>     type='OCRDataset',
        >>>     data_root=ic15_rec_data_root,
        >>>     ann_file='textrecog_test.json',
        >>>     test_mode=True,
        >>>     pipeline=None)
    """

    def __init__(
        self,
        data_root: str,
        dataset_name: str,
        train_anns: Optional[List[Dict]] = [dict(file='textrecog_train.json')],
        val_anns: Optional[List[Dict]] = [],
        test_anns: Optional[List[Dict]] = [dict(file='textrecog_test.json')],
        config_path: str = 'configs/',
    ) -> None:
        super().__init__(
            data_root=data_root,
            task='textrecog',
            dataset_name=dataset_name,
            train_anns=train_anns,
            val_anns=val_anns,
            test_anns=test_anns,
            config_path=config_path)

    def _gen_dataset_config(self) -> str:
        """Generate a full dataset config based on the annotation file
        dictionary.

        Args:
            ann_dict (dict[str, dict(str, str)]): A nested dictionary that maps
                a config variable name (such as icdar2015_textrecog_train) to
                its corresponding annotation information dict. Each dict
                contains following keys:
                - file (str): The path to the annotation file relative to
                data_root.
                - split (str): The split the annotation belongs to. Usually
                it can be 'train', 'val' and 'test'.
                - prefix (str): Affects the prefix of the resulting
                variable in the generated config.

        Returns:
            str: The generated dataset config.
        """
        cfg = ''
        for key_name, ann_dict in self.anns.items():
            cfg += f'\n{key_name} = dict(\n'
            cfg += '    type=\'OCRDataset\',\n'
            cfg += '    data_root=' + f'{self.dataset_name}_{self.task}_data_root,\n'  # noqa: E501
            cfg += f'    ann_file=\'{ann_dict["file"]}\',\n'
            if ann_dict['split'] in ['test', 'val']:
                cfg += '    test_mode=True,\n'
            cfg += '    pipeline=None)\n'
        return cfg

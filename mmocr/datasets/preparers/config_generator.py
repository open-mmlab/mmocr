# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from abc import abstractmethod
from typing import Dict, List, Optional

from mmengine import mkdir_or_exist
from .data_preparer import CFG_GENERATORS


class BaseDatasetConfigGenerator:
    """Base class for dataset config generator.

    Args:
        data_root (str): The root path of the dataset.
        task (str): The task of the dataset.
        dataset_name (str): The name of the dataset.
        overwrite_cfg (bool): Whether to overwrite the dataset config file if
            it already exists. If False, config generator will not generate new
            config for datasets whose configs are already in base.
        train_anns (List[Dict], optional): A list of train annotation files
            to appear in the base configs. Defaults to None.
            Each element is typically a dict with the following fields:
            - ann_file (str): The path to the annotation file relative to
              data_root.
            - dataset_postfix (str, optional): Affects the postfix of the
              resulting variable in the generated config. If specified, the
              dataset variable will be named in the form of
              ``{dataset_name}_{dataset_postfix}_{task}_{split}``. Defaults to
              None.
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
        overwrite_cfg: bool = False,
        train_anns: Optional[List[Dict]] = None,
        val_anns: Optional[List[Dict]] = None,
        test_anns: Optional[List[Dict]] = None,
        config_path: str = 'configs/',
    ) -> None:
        self.config_path = config_path
        self.data_root = data_root
        self.task = task
        self.dataset_name = dataset_name
        self.overwrite_cfg = overwrite_cfg
        self._prepare_anns(train_anns, val_anns, test_anns)

    def _prepare_anns(self, train_anns: Optional[List[Dict]],
                      val_anns: Optional[List[Dict]],
                      test_anns: Optional[List[Dict]]) -> None:
        """Preprocess input arguments and stores these information into
        ``self.anns``.

        ``self.anns`` is a dict that maps the name of a dataset config variable
        to a dict, which contains the following fields:
        - ann_file (str): The path to the annotation file relative to
          data_root.
        - split (str): The split the annotation belongs to. Usually
          it can be 'train', 'val' and 'test'.
        - dataset_postfix (str, optional): Affects the postfix of the
          resulting variable in the generated config. If specified, the
          dataset variable will be named in the form of
          ``{dataset_name}_{dataset_postfix}_{task}_{split}``. Defaults to
          None.
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
                assert 'ann_file' in ann_dict
                if ann_dict.get('dataset_postfix', ''):
                    key = f'{self.dataset_name}_{ann_dict["dataset_postfix"]}_{self.task}_{split}'  # noqa
                else:
                    key = f'{self.dataset_name}_{self.task}_{split}'
                ann_dict['split'] = split
                if key in self.anns:
                    raise ValueError(
                        f'Duplicate dataset variable {key} found! '
                        'Please use different dataset_postfix to avoid '
                        'conflict.')
                self.anns[key] = ann_dict

    def __call__(self) -> None:
        """Generates the base dataset config."""

        dataset_config = self._gen_dataset_config()

        cfg_path = osp.join(self.config_path, self.task, '_base_', 'datasets',
                            f'{self.dataset_name}.py')
        if osp.exists(cfg_path) and not self.overwrite_cfg:
            print(f'{cfg_path} found, skipping.')
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
class TextDetConfigGenerator(BaseDatasetConfigGenerator):
    """Text detection config generator.

    Args:
        data_root (str): The root path of the dataset.
        dataset_name (str): The name of the dataset.
        overwrite_cfg (bool): Whether to overwrite the dataset config file if
            it already exists. If False, config generator will not generate new
            config for datasets whose configs are already in base.
        train_anns (List[Dict], optional): A list of train annotation files
            to appear in the base configs. Defaults to
            ``[dict(file='textdet_train.json', dataset_postfix='')]``.
            Each element is typically a dict with the following fields:
            - ann_file (str): The path to the annotation file relative to
              data_root.
            - dataset_postfix (str, optional): Affects the postfix of the
              resulting variable in the generated config. If specified, the
              dataset variable will be named in the form of
              ``{dataset_name}_{dataset_postfix}_{task}_{split}``. Defaults to
              None.
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
        overwrite_cfg: bool = False,
        train_anns: Optional[List[Dict]] = [
            dict(ann_file='textdet_train.json', dataset_postfix='')
        ],
        val_anns: Optional[List[Dict]] = [],
        test_anns: Optional[List[Dict]] = [
            dict(ann_file='textdet_test.json', dataset_postfix='')
        ],
        config_path: str = 'configs/',
    ) -> None:
        super().__init__(
            data_root=data_root,
            task='textdet',
            overwrite_cfg=overwrite_cfg,
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
                - ann_file (str): The path to the annotation file relative to
                  data_root.
                - dataset_postfix (str, optional): Affects the postfix of the
                  resulting variable in the generated config. If specified, the
                  dataset variable will be named in the form of
                  ``{dataset_name}_{dataset_postfix}_{task}_{split}``. Defaults
                  to None.
                - split (str): The split the annotation belongs to. Usually
                  it can be 'train', 'val' and 'test'.

        Returns:
            str: The generated dataset config.
        """
        cfg = ''
        for key_name, ann_dict in self.anns.items():
            cfg += f'\n{key_name} = dict(\n'
            cfg += '    type=\'OCRDataset\',\n'
            cfg += '    data_root=' + f'{self.dataset_name}_{self.task}_data_root,\n'  # noqa: E501
            cfg += f'    ann_file=\'{ann_dict["ann_file"]}\',\n'
            if ann_dict['split'] == 'train':
                cfg += '    filter_cfg=dict(filter_empty_gt=True, min_size=32),\n'  # noqa: E501
            elif ann_dict['split'] in ['test', 'val']:
                cfg += '    test_mode=True,\n'
            cfg += '    pipeline=None)\n'
        return cfg


@CFG_GENERATORS.register_module()
class TextRecogConfigGenerator(BaseDatasetConfigGenerator):
    """Text recognition config generator.

    Args:
        data_root (str): The root path of the dataset.
        dataset_name (str): The name of the dataset.
        overwrite_cfg (bool): Whether to overwrite the dataset config file if
            it already exists. If False, config generator will not generate new
            config for datasets whose configs are already in base.
        train_anns (List[Dict], optional): A list of train annotation files
            to appear in the base configs. Defaults to
            ``[dict(file='textrecog_train.json'), dataset_postfix='']``.
            Each element is typically a dict with the following fields:
            - ann_file (str): The path to the annotation file relative to
              data_root.
            - dataset_postfix (str, optional): Affects the postfix of the
              resulting variable in the generated config. If specified, the
              dataset variable will be named in the form of
              ``{dataset_name}_{dataset_postfix}_{task}_{split}``. Defaults to
              None.
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
        overwrite_cfg: bool = False,
        train_anns: Optional[List[Dict]] = [
            dict(ann_file='textrecog_train.json', dataset_postfix='')
        ],
        val_anns: Optional[List[Dict]] = [],
        test_anns: Optional[List[Dict]] = [
            dict(ann_file='textrecog_test.json', dataset_postfix='')
        ],
        config_path: str = 'configs/',
    ) -> None:
        super().__init__(
            data_root=data_root,
            task='textrecog',
            overwrite_cfg=overwrite_cfg,
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
                - ann_file (str): The path to the annotation file relative to
                  data_root.
                - dataset_postfix (str, optional): Affects the postfix of the
                  resulting variable in the generated config. If specified, the
                  dataset variable will be named in the form of
                  ``{dataset_name}_{dataset_postfix}_{task}_{split}``. Defaults
                  to None.
                - split (str): The split the annotation belongs to. Usually
                  it can be 'train', 'val' and 'test'.

        Returns:
            str: The generated dataset config.
        """
        cfg = ''
        for key_name, ann_dict in self.anns.items():
            cfg += f'\n{key_name} = dict(\n'
            cfg += '    type=\'OCRDataset\',\n'
            cfg += '    data_root=' + f'{self.dataset_name}_{self.task}_data_root,\n'  # noqa: E501
            cfg += f'    ann_file=\'{ann_dict["ann_file"]}\',\n'
            if ann_dict['split'] in ['test', 'val']:
                cfg += '    test_mode=True,\n'
            cfg += '    pipeline=None)\n'
        return cfg


@CFG_GENERATORS.register_module()
class TextSpottingConfigGenerator(TextDetConfigGenerator):
    """Text spotting config generator.

    Args:
        data_root (str): The root path of the dataset.
        dataset_name (str): The name of the dataset.
        overwrite_cfg (bool): Whether to overwrite the dataset config file if
            it already exists. If False, config generator will not generate new
            config for datasets whose configs are already in base.
        train_anns (List[Dict], optional): A list of train annotation files
            to appear in the base configs. Defaults to
            ``[dict(file='textspotting_train.json', dataset_postfix='')]``.
            Each element is typically a dict with the following fields:
            - ann_file (str): The path to the annotation file relative to
              data_root.
            - dataset_postfix (str, optional): Affects the postfix of the
              resulting variable in the generated config. If specified, the
              dataset variable will be named in the form of
              ``{dataset_name}_{dataset_postfix}_{task}_{split}``. Defaults to
              None.
        val_anns (List[Dict], optional): A list of val annotation files
            to appear in the base configs, similar to ``train_anns``. Defaults
            to [].
        test_anns (List[Dict], optional): A list of test annotation files
            to appear in the base configs, similar to ``train_anns``. Defaults
            to ``[dict(file='textspotting_test.json')]``.
        config_path (str): Path to the configs. Defaults to 'configs/'.
    """

    def __init__(
        self,
        data_root: str,
        dataset_name: str,
        overwrite_cfg: bool = False,
        train_anns: Optional[List[Dict]] = [
            dict(ann_file='textspotting_train.json', dataset_postfix='')
        ],
        val_anns: Optional[List[Dict]] = [],
        test_anns: Optional[List[Dict]] = [
            dict(ann_file='textspotting_test.json', dataset_postfix='')
        ],
        config_path: str = 'configs/',
    ) -> None:
        BaseDatasetConfigGenerator.__init__(
            self,
            data_root=data_root,
            task='textspotting',
            overwrite_cfg=overwrite_cfg,
            dataset_name=dataset_name,
            train_anns=train_anns,
            val_anns=val_anns,
            test_anns=test_anns,
            config_path=config_path,
        )

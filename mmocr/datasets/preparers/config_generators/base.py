# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from abc import abstractmethod
from typing import Dict, List, Optional

from mmengine import mkdir_or_exist


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

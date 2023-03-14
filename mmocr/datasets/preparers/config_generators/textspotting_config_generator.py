# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional

from mmocr.registry import CFG_GENERATORS
from .base import BaseDatasetConfigGenerator
from .textdet_config_generator import TextDetConfigGenerator


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

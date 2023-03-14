# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import unittest

from mmengine import Config

from mmocr.datasets.preparers import DatasetPreparer
from mmocr.datasets.preparers.data_preparer import (CFG_GENERATORS,
                                                    DATA_DUMPERS,
                                                    DATA_GATHERERS,
                                                    DATA_OBTAINERS,
                                                    DATA_PACKERS, DATA_PARSERS)


class Fake:

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return None, None


DATA_OBTAINERS.register_module(module=Fake)
DATA_GATHERERS.register_module(module=Fake)
DATA_PARSERS.register_module(module=Fake)
DATA_DUMPERS.register_module(module=Fake)
DATA_PACKERS.register_module(module=Fake)
CFG_GENERATORS.register_module(module=Fake)


class TestDataPreparer(unittest.TestCase):

    def _create_config(self):
        cfg_path = 'config.py'
        cfg = ''
        cfg += "data_root = ''\n"
        cfg += 'train_preparer=dict(\n'
        cfg += '    obtainer=dict(type="Fake"),\n'
        cfg += '    gatherer=dict(type="Fake"),\n'
        cfg += '    parser=dict(type="Fake"),\n'
        cfg += '    packer=dict(type="Fake"),\n'
        cfg += '    dumper=dict(type="Fake"),\n'
        cfg += ')\n'
        cfg += 'test_preparer=dict(\n'
        cfg += '    obtainer=dict(type="Fake"),\n'
        cfg += ')\n'
        cfg += 'cfg_generator=dict(type="Fake")\n'
        cfg += f"delete = ['{cfg_path}']\n"

        with open(cfg_path, 'w') as f:
            f.write(cfg)
        return cfg_path

    def test_dataset_preparer(self):
        cfg_path = self._create_config()
        cfg = Config.fromfile(cfg_path)
        preparer = DatasetPreparer.from_file(cfg)
        preparer.run()
        self.assertFalse(osp.exists(cfg_path))

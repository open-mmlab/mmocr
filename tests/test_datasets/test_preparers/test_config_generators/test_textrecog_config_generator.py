# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import unittest

from mmocr.datasets.preparers import TextRecogConfigGenerator


class TestTextRecogConfigGenerator(unittest.TestCase):

    def setUp(self) -> None:
        self.root = tempfile.TemporaryDirectory()

    def test_textrecog_config_generator(self):
        config_generator = TextRecogConfigGenerator(
            data_root=self.root.name,
            dataset_name='dummy',
            train_anns=[
                dict(ann_file='textrecog_train.json', dataset_postfix='')
            ],
            val_anns=[],
            test_anns=[
                dict(ann_file='textrecog_test.json', dataset_postfix='fake')
            ],
            config_path=self.root.name,
        )
        cfg_path = osp.join(self.root.name, 'textrecog', '_base_', 'datasets',
                            'dummy.py')
        config_generator()
        self.assertTrue(osp.exists(cfg_path))
        f = open(cfg_path, 'r')
        lines = ''.join(f.readlines())

        self.assertEquals(lines,
                          (f"dummy_textrecog_data_root = '{self.root.name}'\n"
                           '\n'
                           'dummy_textrecog_train = dict(\n'
                           "    type='OCRDataset',\n"
                           '    data_root=dummy_textrecog_data_root,\n'
                           "    ann_file='textrecog_train.json',\n"
                           '    pipeline=None)\n'
                           '\n'
                           'dummy_fake_textrecog_test = dict(\n'
                           "    type='OCRDataset',\n"
                           '    data_root=dummy_textrecog_data_root,\n'
                           "    ann_file='textrecog_test.json',\n"
                           '    test_mode=True,\n'
                           '    pipeline=None)\n'))
        with self.assertRaises(ValueError):
            TextRecogConfigGenerator(
                data_root=self.root.name,
                dataset_name='dummy',
                train_anns=[
                    dict(ann_file='textrecog_train.json', dataset_postfix='1'),
                    dict(
                        ann_file='textrecog_train_1.json', dataset_postfix='1')
                ],
                config_path=self.root.name,
            )

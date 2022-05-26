# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import sys
from unittest import TestCase

from mmengine import DefaultScope

from mmocr.utils import register_all_modules


class TestSetupEnv(TestCase):

    def test_register_all_modules(self):
        from mmocr.registry import DATASETS

        # not init default scope
        sys.modules.pop('mmocr.datasets', None)
        sys.modules.pop('mmocr.datasets.ocr_dataset', None)
        DATASETS._module_dict.pop('OCRDataset', None)
        self.assertFalse('OCRDataset' in DATASETS.module_dict)
        register_all_modules(init_default_scope=False)
        self.assertTrue('OCRDataset' in DATASETS.module_dict)

        # init default scope
        sys.modules.pop('mmocr.datasets')
        sys.modules.pop('mmocr.datasets.ocr_dataset')
        DATASETS._module_dict.pop('OCRDataset', None)
        self.assertFalse('OCRDataset' in DATASETS.module_dict)
        register_all_modules(init_default_scope=True)
        self.assertTrue('OCRDataset' in DATASETS.module_dict)
        self.assertEqual(DefaultScope.get_current_instance().scope_name,
                         'mmocr')

        # init default scope when another scope is init
        name = f'test-{datetime.datetime.now()}'
        DefaultScope.get_instance(name, scope_name='test')
        with self.assertWarnsRegex(
                Warning, 'The current default scope "test" is not "mmocr"'):
            register_all_modules(init_default_scope=True)

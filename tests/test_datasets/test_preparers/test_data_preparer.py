# Copyright (c) OpenMMLab. All rights reserved.
import unittest

from mmocr.datasets.preparers.data_preparer import DatasetPreparer


class TestDataPreparer(unittest.TestCase):

    def setUp(self) -> None:
        self.cfg_path = 'tests/data/preparer'
        self.dataset_name = 'dummy'

    def test_dataset_preparer(self):
        preparer = DatasetPreparer(self.cfg_path, self.dataset_name, 'textdet')
        preparer()

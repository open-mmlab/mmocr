# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
import tempfile
import unittest

from mmocr.datasets.preparers.dumpers import (JsonDumper,
                                              WildreceiptOpensetDumper)


class TestDumpers(unittest.TestCase):

    def setUp(self) -> None:
        self.root = tempfile.TemporaryDirectory()

    def test_json_dumpers(self):
        task, split = 'textdet', 'train'
        fake_data = dict(
            metainfo=dict(
                dataset_type='TextDetDataset',
                task_name='textdet',
                category=[dict(id=0, name='text')]))

        dumper = JsonDumper(task, dataset_name='test')
        dumper.dump(fake_data, self.root.name, split)
        with open(osp.join(self.root.name, f'{task}_{split}.json'), 'r') as f:
            data = json.load(f)
        self.assertEqual(data, fake_data)

    def test_wildreceipt_dumper(self):
        task, split = 'kie', 'train'
        fake_data = ['test1', 'test2']

        dumper = WildreceiptOpensetDumper(task)
        dumper.dump(fake_data, self.root.name, split)
        with open(osp.join(self.root.name, f'openset_{split}.txt'), 'r') as f:
            data = f.read().splitlines()
        self.assertEqual(data, fake_data)

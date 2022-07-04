# Copyright (c) OpenMMLab. All rights reserved.
import json
from unittest import TestCase

from mmocr.datasets import LineJsonParser, LineStrParser


class TestParser(TestCase):

    def test_line_json_parser(self):
        parser = LineJsonParser()
        line = json.dumps(dict(filename='test.jpg', text='mmocr'))
        data = parser(line)
        self.assertEqual(data['filename'], 'test.jpg')
        self.assertEqual(data['text'], 'mmocr')

    def test_line_str_parser(self):
        parser = LineStrParser()
        line = 'test.jpg mmocr'
        data = parser(line)
        self.assertEqual(data['filename'], 'test.jpg')
        self.assertEqual(data['text'], 'mmocr')

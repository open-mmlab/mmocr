# Copyright (c) OpenMMLab. All rights reserved.
import copy
import unittest

from mmengine.data import LabelData

from mmocr.core import TextRecogDataSample
from mmocr.evaluation import CharMetric, OneMinusNEDMetric, WordMetric


class TestWordMetric(unittest.TestCase):

    def setUp(self):
        # prepare gt hello HELLO $HELLO$
        gt1 = {
            'data_sample': {
                'height': 32,
                'width': 100,
                'instances': [{
                    'text': 'hello'
                }]
            }
        }
        gt2 = {
            'data_sample': {
                'height': 32,
                'width': 100,
                'instances': [{
                    'text': 'HELLO'
                }]
            }
        }
        gt3 = {
            'data_sample': {
                'height': 32,
                'width': 100,
                'instances': [{
                    'text': '$HELLO$'
                }]
            }
        }
        self.gt = [gt1, gt2, gt3]
        # prepare pred
        pred_data_sample = TextRecogDataSample()
        pred_text = LabelData()
        pred_text.item = 'hello'
        pred_data_sample.pred_text = pred_text

        self.pred = [
            pred_data_sample,
            copy.deepcopy(pred_data_sample),
            copy.deepcopy(pred_data_sample),
        ]

    def test_word_acc_metric(self):
        metric = WordMetric(mode='exact')
        metric.process(self.gt, self.pred)
        eval_res = metric.evaluate(size=3)
        self.assertAlmostEqual(eval_res['recog/word_acc'], 1. / 3, 4)

    def test_word_acc_ignore_case_metric(self):
        metric = WordMetric(mode='ignore_case')
        metric.process(self.gt, self.pred)
        eval_res = metric.evaluate(size=3)
        self.assertAlmostEqual(eval_res['recog/word_acc_ignore_case'], 2. / 3,
                               4)

    def test_word_acc_ignore_case_symbol_metric(self):
        metric = WordMetric(mode='ignore_case_symbol')
        metric.process(self.gt, self.pred)
        eval_res = metric.evaluate(size=3)
        self.assertEqual(eval_res['recog/word_acc_ignore_case_symbol'], 1.0)

    def test_all_metric(self):
        metric = WordMetric(
            mode=['exact', 'ignore_case', 'ignore_case_symbol'])
        metric.process(self.gt, self.pred)
        eval_res = metric.evaluate(size=3)
        self.assertAlmostEqual(eval_res['recog/word_acc'], 1. / 3, 4)
        self.assertAlmostEqual(eval_res['recog/word_acc_ignore_case'], 2. / 3,
                               4)
        self.assertEqual(eval_res['recog/word_acc_ignore_case_symbol'], 1.0)


class TestCharMetric(unittest.TestCase):

    def setUp(self):
        # prepare gt
        gt1 = {
            'data_sample': {
                'height': 32,
                'width': 100,
                'instances': [{
                    'text': 'hello'
                }]
            }
        }
        gt2 = {
            'data_sample': {
                'height': 32,
                'width': 100,
                'instances': [{
                    'text': 'HELLO'
                }]
            }
        }
        self.gt = [gt1, gt2]
        # prepare pred
        pred_data_sample1 = TextRecogDataSample()
        pred_text = LabelData()
        pred_text.item = 'helL'
        pred_data_sample1.pred_text = pred_text

        pred_data_sample2 = TextRecogDataSample()
        pred_text = LabelData()
        pred_text.item = 'HEL'
        pred_data_sample2.pred_text = pred_text

        self.pred = [pred_data_sample1, pred_data_sample2]

    def test_char_recall_precision_metric(self):
        metric = CharMetric()
        metric.process(self.gt, self.pred)
        eval_res = metric.evaluate(size=2)
        self.assertEqual(eval_res['recog/char_recall'], 0.7)
        self.assertEqual(eval_res['recog/char_precision'], 1)


class TestOneMinusNED(unittest.TestCase):

    def setUp(self):
        # prepare gt
        gt1 = {
            'data_sample': {
                'height': 32,
                'width': 100,
                'instances': [{
                    'text': 'hello'
                }]
            }
        }
        gt2 = {
            'data_sample': {
                'height': 32,
                'width': 100,
                'instances': [{
                    'text': 'HELLO'
                }]
            }
        }
        self.gt = [gt1, gt2]
        # prepare pred
        pred_data_sample1 = TextRecogDataSample()
        pred_text = LabelData()
        pred_text.item = 'pred_helL'
        pred_data_sample1.pred_text = pred_text

        pred_data_sample2 = TextRecogDataSample()
        pred_text = LabelData()
        pred_text.item = 'HEL'
        pred_data_sample2.pred_text = pred_text

        self.pred = [pred_data_sample1, pred_data_sample2]

    def test_one_minus_ned_metric(self):
        metric = OneMinusNEDMetric()
        metric.process(self.gt, self.pred)
        eval_res = metric.evaluate(size=2)
        self.assertEqual(eval_res['recog/1-N.E.D'], 0.4875)

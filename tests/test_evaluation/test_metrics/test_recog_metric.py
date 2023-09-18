# Copyright (c) OpenMMLab. All rights reserved.
import unittest

from mmengine.structures import LabelData

from mmocr.evaluation import CharMetric, OneMinusNEDMetric, WordMetric
from mmocr.structures import TextRecogDataSample


class TestWordMetric(unittest.TestCase):

    def setUp(self):

        self.pred = []
        data_sample = TextRecogDataSample()
        pred_text = LabelData()
        pred_text.item = 'hello'
        data_sample.pred_text = pred_text
        gt_text = LabelData()
        gt_text.item = 'hello'
        data_sample.gt_text = gt_text
        self.pred.append(data_sample)
        data_sample = TextRecogDataSample()
        pred_text = LabelData()
        pred_text.item = 'hello'
        data_sample.pred_text = pred_text
        gt_text = LabelData()
        gt_text.item = 'HELLO'
        data_sample.gt_text = gt_text
        self.pred.append(data_sample)
        data_sample = TextRecogDataSample()
        pred_text = LabelData()
        pred_text.item = 'hello'
        data_sample.pred_text = pred_text
        gt_text = LabelData()
        gt_text.item = '$HELLO$'
        data_sample.gt_text = gt_text
        self.pred.append(data_sample)

    def test_word_acc_metric(self):
        metric = WordMetric(mode='exact')
        metric.process(None, self.pred)
        eval_res = metric.evaluate(size=3)
        self.assertAlmostEqual(eval_res['WordMetric/accuracy'], 1. / 3, 4)

    def test_word_acc_ignore_case_metric(self):
        metric = WordMetric(mode='ignore_case')
        metric.process(None, self.pred)
        eval_res = metric.evaluate(size=3)
        self.assertAlmostEqual(eval_res['WordMetric/ignore_case_accuracy'],
                               2. / 3, 4)

    def test_word_acc_ignore_case_symbol_metric(self):
        metric = WordMetric(mode='ignore_case_symbol')
        metric.process(None, self.pred)
        eval_res = metric.evaluate(size=3)
        self.assertEqual(eval_res['WordMetric/ignore_case_symbol_accuracy'],
                         1.0)

    def test_all_metric(self):
        metric = WordMetric(
            mode=['exact', 'ignore_case', 'ignore_case_symbol'])
        metric.process(None, self.pred)
        eval_res = metric.evaluate(size=3)
        self.assertAlmostEqual(eval_res['WordMetric/accuracy'], 1. / 3, 4)
        self.assertAlmostEqual(eval_res['WordMetric/ignore_case_accuracy'],
                               2. / 3, 4)
        self.assertEqual(eval_res['WordMetric/ignore_case_symbol_accuracy'],
                         1.0)


class TestCharMetric(unittest.TestCase):

    def setUp(self):
        self.pred = []
        data_sample = TextRecogDataSample()
        pred_text = LabelData()
        pred_text.item = 'helL'
        data_sample.pred_text = pred_text
        gt_text = LabelData()
        gt_text.item = 'hello'
        data_sample.gt_text = gt_text
        self.pred.append(data_sample)
        data_sample = TextRecogDataSample()
        pred_text = LabelData()
        pred_text.item = 'HEL'
        data_sample.pred_text = pred_text
        gt_text = LabelData()
        gt_text.item = 'HELLO'
        data_sample.gt_text = gt_text
        self.pred.append(data_sample)

    def test_char_recall_precision_metric(self):
        metric = CharMetric()
        metric.process(None, self.pred)
        eval_res = metric.evaluate(size=2)
        self.assertEqual(eval_res['CharMetric/recall'], 0.7)
        self.assertEqual(eval_res['CharMetric/precision'], 1)


class TestOneMinusNED(unittest.TestCase):

    def setUp(self):
        self.pred = []
        data_sample = TextRecogDataSample()
        pred_text = LabelData()
        pred_text.item = 'pred_helL'
        data_sample.pred_text = pred_text
        gt_text = LabelData()
        gt_text.item = 'hello'
        data_sample.gt_text = gt_text
        self.pred.append(data_sample)
        data_sample = TextRecogDataSample()
        pred_text = LabelData()
        pred_text.item = 'HEL'
        data_sample.pred_text = pred_text
        gt_text = LabelData()
        gt_text.item = 'HELLO'
        data_sample.gt_text = gt_text
        self.pred.append(data_sample)

    def test_one_minus_ned_metric(self):
        metric = OneMinusNEDMetric()
        metric.process(None, self.pred)
        eval_res = metric.evaluate(size=2)
        self.assertEqual(eval_res['OneMinusNEDMetric/1-N.E.D'], 0.4875)

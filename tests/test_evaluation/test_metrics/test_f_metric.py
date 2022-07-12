# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch
from mmengine import InstanceData

from mmocr.core import KIEDataSample
from mmocr.evaluation import F1Metric


class TestF1Metric(unittest.TestCase):

    def test_init(self):
        with self.assertRaises(AssertionError):
            F1Metric(num_classes='3')

        with self.assertRaises(AssertionError):
            F1Metric(num_classes=3, ignored_classes=[1], cared_classes=[0])

        with self.assertRaises(AssertionError):
            F1Metric(num_classes=3, ignored_classes=1)

        with self.assertRaises(AssertionError):
            F1Metric(num_classes=2, mode=1)

        with self.assertRaises(AssertionError):
            F1Metric(num_classes=1, mode='1')

    def test_macro_f1(self):
        mode = 'macro'
        gts_cases = [
            [
                dict(
                    data_sample=KIEDataSample(
                        gt_instances=InstanceData(
                            labels=torch.LongTensor([0, 1, 4]))))
            ],
            [
                dict(
                    data_sample=KIEDataSample(
                        gt_instances=InstanceData(
                            labels=torch.LongTensor([0, 1])))),
                dict(
                    data_sample=KIEDataSample(
                        gt_instances=InstanceData(
                            labels=torch.LongTensor([4]))))
            ],
        ]
        preds_cases = [[
            KIEDataSample(
                pred_instances=InstanceData(
                    labels=torch.LongTensor([0, 1, 2])))
        ],
                       [
                           KIEDataSample(
                               pred_instances=InstanceData(
                                   labels=torch.LongTensor([0, 1]))),
                           KIEDataSample(
                               pred_instances=InstanceData(
                                   labels=torch.LongTensor([2])))
                       ]]

        # num_classes < the maximum label index
        metric = F1Metric(num_classes=3, ignored_classes=[1])
        metric.process(gts_cases[0], preds_cases[0])
        with self.assertRaises(AssertionError):
            metric.evaluate(size=1)

        for gts, preds in zip(gts_cases, preds_cases):
            metric = F1Metric(num_classes=5, mode=mode)
            metric.process(gts, preds)
            result = metric.evaluate(size=len(gts))
            self.assertAlmostEqual(result['kie/macro_f1'], 0.4)

            # Test ignored_classes
            metric = F1Metric(num_classes=5, ignored_classes=[1], mode=mode)
            metric.process(gts, preds)
            result = metric.evaluate(size=len(gts))
            self.assertAlmostEqual(result['kie/macro_f1'], 0.25)

            # Test cared_classes
            metric = F1Metric(
                num_classes=5, cared_classes=[0, 2, 3, 4], mode=mode)
            metric.process(gts, preds)
            result = metric.evaluate(size=len(gts))
            self.assertAlmostEqual(result['kie/macro_f1'], 0.25)

    def test_micro_f1(self):
        mode = 'micro'
        gts_cases = [[
            dict(
                data_sample=KIEDataSample(
                    gt_instances=InstanceData(
                        labels=torch.LongTensor([0, 1, 0, 1, 2]))))
        ],
                     [
                         dict(
                             data_sample=KIEDataSample(
                                 gt_instances=InstanceData(
                                     labels=torch.LongTensor([0, 1, 2])))),
                         dict(
                             data_sample=KIEDataSample(
                                 gt_instances=InstanceData(
                                     labels=torch.LongTensor([0, 1]))))
                     ]]
        preds_cases = [[
            KIEDataSample(
                pred_instances=InstanceData(
                    labels=torch.LongTensor([0, 1, 2, 2, 0])))
        ],
                       [
                           KIEDataSample(
                               pred_instances=InstanceData(
                                   labels=torch.LongTensor([0, 1, 0]))),
                           KIEDataSample(
                               pred_instances=InstanceData(
                                   labels=torch.LongTensor([2, 2])))
                       ]]

        # num_classes < the maximum label index
        metric = F1Metric(num_classes=1, ignored_classes=[0], mode=mode)
        metric.process(gts_cases[0], preds_cases[0])
        with self.assertRaises(AssertionError):
            metric.evaluate(size=1)

        for gts, preds in zip(gts_cases, preds_cases):
            # class 0: tp: 1, fp: 1, fn: 1
            # class 1: tp: 1, fp: 1, fn: 0
            # class 2: tp: 0, fp: 1, fn: 2
            # overall: tp: 2, fp: 3, fn: 3
            # f1: 0.4

            metric = F1Metric(num_classes=3, mode=mode)
            metric.process(gts, preds)
            result = metric.evaluate(size=len(gts))
            self.assertAlmostEqual(result['kie/micro_f1'], 0.4, delta=0.01)

            metric = F1Metric(num_classes=5, mode=mode)
            metric.process(gts, preds)
            result = metric.evaluate(size=len(gts))
            self.assertAlmostEqual(result['kie/micro_f1'], 0.4, delta=0.01)

            # class 0: tp: 1, fp: 1, fn: 1
            # class 2: tp: 0, fp: 1, fn: 2
            # overall: tp: 1, fp: 2, fn: 3
            # f1: 0.285

            metric = F1Metric(num_classes=5, ignored_classes=[1], mode=mode)
            metric.process(gts, preds)
            result = metric.evaluate(size=len(gts))
            self.assertAlmostEqual(result['kie/micro_f1'], 0.285, delta=0.001)

            metric = F1Metric(
                num_classes=5, cared_classes=[0, 2, 3, 4], mode=mode)
            metric.process(gts, preds)
            result = metric.evaluate(size=len(gts))
            self.assertAlmostEqual(result['kie/micro_f1'], 0.285, delta=0.001)

    def test_arguments(self):
        mode = ['micro', 'macro']
        gts = [
            dict(
                data_sample=KIEDataSample(
                    gt_instances=InstanceData(
                        test_labels=torch.LongTensor([0, 1, 0, 1, 2]))))
        ]
        preds = [
            KIEDataSample(
                pred_instances=InstanceData(
                    test_labels=torch.LongTensor([0, 1, 2, 2, 0])))
        ]

        # class 0: tp: 1, fp: 1, fn: 1
        # class 1: tp: 1, fp: 1, fn: 0
        # class 2: tp: 0, fp: 1, fn: 2
        # overall: tp: 2, fp: 3, fn: 3
        # micro_f1: 0.4
        # macro_f1:

        metric = F1Metric(num_classes=3, mode=mode, key='test_labels')
        metric.process(gts, preds)
        result = metric.evaluate(size=1)
        self.assertAlmostEqual(result['kie/micro_f1'], 0.4, delta=0.01)
        self.assertAlmostEqual(result['kie/macro_f1'], 0.39, delta=0.01)

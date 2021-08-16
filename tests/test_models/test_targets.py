# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmocr.datasets.pipelines.textdet_targets.dbnet_targets import DBNetTargets


def test_invalid_polys():

    dbtarget = DBNetTargets()

    poly = np.array([[256.1229216, 347.17471155], [257.63126133, 347.0069367],
                     [257.70317729, 347.65337423],
                     [256.19488113, 347.82114909]])

    assert dbtarget.invalid_polygon(poly)

    poly = np.array([[570.34735492,
                      335.00214526], [570.99778839, 335.00327318],
                     [569.69077318, 338.47009908],
                     [569.04038393, 338.46894904]])
    assert dbtarget.invalid_polygon(poly)

    poly = np.array([[481.18343777,
                      305.03190065], [479.88478587, 305.10684512],
                     [479.90976971, 305.53968843], [480.99197962,
                                                    305.4772347]])
    assert dbtarget.invalid_polygon(poly)

    poly = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
    assert dbtarget.invalid_polygon(poly)

    poly = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    assert not dbtarget.invalid_polygon(poly)

# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch
from numpy.testing import assert_array_equal

from mmocr.apis.utils import tensor2grayimgs


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
def test_tensor2grayimgs():

    # test tensor obj
    with pytest.raises(AssertionError):
        tensor = np.random.rand(2, 3, 3)
        tensor2grayimgs(tensor)

    # test tensor ndim
    with pytest.raises(AssertionError):
        tensor = torch.randn(2, 3, 3)
        tensor2grayimgs(tensor)

    # test tensor dim-1
    with pytest.raises(AssertionError):
        tensor = torch.randn(2, 3, 5, 5)
        tensor2grayimgs(tensor)

    # test mean length
    with pytest.raises(AssertionError):
        tensor = torch.randn(2, 1, 5, 5)
        tensor2grayimgs(tensor, mean=(1, 1, 1))

    # test std length
    with pytest.raises(AssertionError):
        tensor = torch.randn(2, 1, 5, 5)
        tensor2grayimgs(tensor, std=(1, 1, 1))

    tensor = torch.randn(2, 1, 5, 5)
    gts = [t.squeeze(0).cpu().numpy().astype(np.uint8) for t in tensor]
    outputs = tensor2grayimgs(tensor, mean=(0, ), std=(1, ))
    for gt, output in zip(gts, outputs):
        assert_array_equal(gt, output)

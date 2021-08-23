# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mmcv.cnn.bricks import ConvModule

from mmocr.utils import revert_sync_batchnorm


def test_revert_sync_batchnorm():
    conv_syncbn = ConvModule(3, 8, 2, norm_cfg=dict(type='SyncBN')).to('cpu')
    conv_syncbn.train()
    x = torch.randn(1, 3, 10, 10)
    # Will raise an ValueError saying SyncBN does not run on CPU
    with pytest.raises(ValueError):
        y = conv_syncbn(x)
    conv_bn = revert_sync_batchnorm(conv_syncbn)
    y = conv_bn(x)
    assert y.shape == (1, 8, 9, 9)
    assert conv_bn.training == conv_syncbn.training
    conv_syncbn.eval()
    conv_bn = revert_sync_batchnorm(conv_syncbn)
    assert conv_bn.training == conv_syncbn.training

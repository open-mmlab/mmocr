# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmocr.models.textrecog.fusers import ABIFuser


def test_base_alignment():
    model = ABIFuser(d_model=512, num_chars=90, max_seq_len=40)
    l_feat = torch.randn(1, 40, 512)
    v_feat = torch.randn(1, 40, 512)
    result = model(l_feat, v_feat)
    assert result['logits'].shape == torch.Size([1, 40, 90])

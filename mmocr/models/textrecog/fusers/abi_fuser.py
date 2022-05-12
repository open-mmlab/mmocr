# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.runner import BaseModule

from mmocr.registry import MODELS


@MODELS.register_module()
class ABIFuser(BaseModule):
    """Mix and align visual feature and linguistic feature Implementation of
    language model of `ABINet <https://arxiv.org/abs/1910.04396>`_.

    Args:
        d_model (int): Hidden size of input.
        max_seq_len (int): Maximum text sequence length :math:`T`.
        num_chars (int): Number of text characters :math:`C`.
        init_cfg (dict): Specifies the initialization method for model layers.
    """

    def __init__(self,
                 d_model=512,
                 max_seq_len=40,
                 num_chars=90,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)

        self.max_seq_len = max_seq_len + 1  # additional stop token
        self.w_att = nn.Linear(2 * d_model, d_model)
        self.cls = nn.Linear(d_model, num_chars)

    def forward(self, l_feature, v_feature):
        """
        Args:
            l_feature: (N, T, E) where T is length, N is batch size and
                d is dim of model.
            v_feature: (N, T, E) shape the same as l_feature.

        Returns:
            A dict with key ``logits``
            The logits of shape (N, T, C) where N is batch size, T is length
                and C is the number of characters.
        """
        f = torch.cat((l_feature, v_feature), dim=2)
        f_att = torch.sigmoid(self.w_att(f))
        output = f_att * v_feature + (1 - f_att) * l_feature

        logits = self.cls(output)  # (N, T, C)

        return {'logits': logits}

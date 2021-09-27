# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.runner import BaseModule

from mmocr.models.builder import FUSERS


@FUSERS.register_module()
class BaseAlignment(BaseModule):

    def __init__(self,
                 d_model=512,
                 max_seq_len=40,
                 num_chars=90,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.max_seq_len = max_seq_len + 1  # additional stop token
        self.w_att = nn.Linear(2 * d_model, d_model)
        self.cls = nn.Linear(d_model, num_chars)

    def forward(self, l_feature, v_feature):
        """
        Args:
            l_feature: (N, T, E) where T is length, N is batch size and
                d is dim of model
            v_feature: (N, T, E) shape the same as l_feature

        Returns:
            The logits of shape (N, T, C) where N is batch size, T is length
                and C is the number of characters.
        """
        f = torch.cat((l_feature, v_feature), dim=2)
        f_att = torch.sigmoid(self.w_att(f))
        output = f_att * v_feature + (1 - f_att) * l_feature

        logits = self.cls(output)  # (N, T, C)
        # pt_lengths = self._get_length(logits)

        return logits

# ------------------------------------------------------------------------------
# Adapted from https://github.com/lonePatient/BERT-NER-Pytorch
# Original licence: Copyright (c) 2020 Weitang Liu, under the MIT License.
# ------------------------------------------------------------------------------

import math

import torch
import torch.nn as nn
from mmcv.cnn import ACTIVATION_LAYERS


@ACTIVATION_LAYERS.register_module()
class GeluNew(nn.Module):
    """Implementation of the gelu activation function currently in Google Bert
    repo (identical to OpenAI GPT).

    Also see https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: Activated tensor.
        """
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

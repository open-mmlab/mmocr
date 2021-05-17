# ------------------------------------------------------------------------------
# Adapted from https://github.com/lonePatient/BERT-NER-Pytorch
# Original licence: Copyright (c) 2020 Weitang Liu, under the MIT License.
# ------------------------------------------------------------------------------

import math

import torch
from mmcv.cnn import Swish


def gelu(x):
    """Original Implementation of the gelu activation function in Google Bert
    repo when initially created. For information: OpenAI GPT's gelu is slightly
    different (and gives slightly different results):

    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
    (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """Implementation of the gelu activation function currently in Google Bert
    repo (identical to OpenAI GPT).

    Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(
        math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


ACT2FN = {
    'gelu': gelu,
    'relu': torch.nn.functional.relu,
    'swish': Swish,
    'gelu_new': gelu_new
}

# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class Focal loss implementation.

    Args:
        gamma (float): The larger the gamma, the smaller
            the loss weight of easier samples.
        weight (float): A manual rescaling weight given to each
            class.
        ignore_index (int): Specifies a target value that is ignored
            and does not contribute to the input gradient.
    """

    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        logit = F.log_softmax(input, dim=1)
        pt = torch.exp(logit)
        logit = (1 - pt)**self.gamma * logit
        loss = F.nll_loss(
            logit, target, self.weight, ignore_index=self.ignore_index)
        return loss

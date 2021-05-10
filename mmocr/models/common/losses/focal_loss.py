import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class Focal loss implementation.

    Args:
        gamma (float): Hyper-parameter.
        weight (float): Hyper-parameter.
        ignore index (int): Ignore index in label.
    """

    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt)**self.gamma * logpt
        loss = F.nll_loss(
            logpt, target, self.weight, ignore_index=self.ignore_index)
        return loss

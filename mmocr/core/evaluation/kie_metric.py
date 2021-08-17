# Copyright (c) OpenMMLab. All rights reserved.
import torch


def compute_f1_score(preds, gts, ignores=[]):
    """Compute the F1-score of prediction.

    Args:
        preds (Tensor): The predicted probability NxC map
            with N and C being the sample number and class
            number respectively.
        gts (Tensor): The ground truth vector of size N.
        ignores (list): The index set of classes that are ignored when
            reporting results.
            Note: all samples are participated in computing.

     Returns:
        The numpy list of f1-scores of valid classes.
    """
    C = preds.size(1)
    classes = torch.LongTensor(sorted(set(range(C)) - set(ignores)))
    hist = torch.bincount(
        gts * C + preds.argmax(1), minlength=C**2).view(C, C).float()
    diag = torch.diag(hist)
    recalls = diag / hist.sum(1).clamp(min=1)
    precisions = diag / hist.sum(0).clamp(min=1)
    f1 = 2 * recalls * precisions / (recalls + precisions).clamp(min=1e-8)
    return f1[classes].cpu().numpy()

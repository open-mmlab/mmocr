import torch
import torch.nn as nn
import torch.nn.functional as F


class DotProductAttentionLayer(nn.Module):

    def __init__(self, dim_model=None):
        super().__init__()

        self.scale = dim_model**-0.5 if dim_model is not None else 1.

    def forward(self, query, key, value, mask=None):
        n, seq_len = mask.size()
        logits = torch.matmul(query.permute(0, 2, 1), key) * self.scale

        if mask is not None:
            mask = mask.view(n, 1, seq_len)
            logits = logits.masked_fill(mask, float('-inf'))

        weights = F.softmax(logits, dim=2)

        glimpse = torch.matmul(weights, value.transpose(1, 2))

        glimpse = glimpse.permute(0, 2, 1).contiguous()

        return glimpse

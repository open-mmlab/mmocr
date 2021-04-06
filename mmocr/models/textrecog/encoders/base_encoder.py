import torch.nn as nn

from mmocr.models.builder import ENCODERS


@ENCODERS.register_module()
class BaseEncoder(nn.Module):
    """Base Encoder class for text recognition."""

    def init_weights(self):
        pass

    def forward(self, feat, **kwargs):
        return feat

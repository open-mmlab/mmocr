import torch.nn as nn

from mmocr.models.builder import PREPROCESSOR


@PREPROCESSOR.register_module()
class BasePreprocessor(nn.Module):
    """Base Preprocessor class for text recognition."""

    def init_weights(self):
        pass

    def forward(self, x, **kwargs):
        return x

from mmcv.runner import BaseModule

from mmocr.models.builder import ENCODERS


@ENCODERS.register_module()
class BaseEncoder(BaseModule):
    """Base Encoder class for text recognition."""

    def init_weights(self):
        pass

    def forward(self, feat, **kwargs):
        return feat

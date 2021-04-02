from mmocr.models.builder import ENCODERS
from .base_encoder import BaseEncoder


@ENCODERS.register_module()
class TFEncoder(BaseEncoder):
    """Encode 2d feature map to 1d sequence."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, feat, img_metas=None):
        n, c, _, _ = feat.size()
        enc_output = feat.view(n, c, -1).transpose(2, 1).contiguous()

        return enc_output

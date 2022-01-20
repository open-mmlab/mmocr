from mmcv.cnn import ConvModule
from mmcv.runner import Sequential

from mmocr.models.builder import ENCODERS
from mmocr.models.textrecog.decoders import CRNNDecoder
from .base_encoder import BaseEncoder


@ENCODERS.register_module()
class ABCRecogEncoder(BaseEncoder):
    """Implement encoder for ABCNet's recognition branch, see `ABCNet.

    <https://arxiv.org/pdf/2002.10200.pdf>`_.

    Args:
        num_channels (int): Number of channels :math:`E` of input vector.
        init_cfg (dict): Specifies the initialization method for model layers.
    """

    def __init__(self,
                 num_channels=1,
                 init_cfg=dict(type='Kaiming', layer='Conv'),
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.convs = Sequential(*[
            ConvModule(
                num_channels,
                num_channels,
                3,
                stride=(2, 1),
                norm_cfg=dict(type='GN', num_groups=32),
                act_cfg=dict(type='ReLU')) for _ in range(2)
        ])
        # Warning: inconsistent with the implementation in adelaidet
        self.crnn = CRNNDecoder(num_channels, num_channels, rnn_flag=True)

    def forward(self, feat, img_metas=None):
        """
        Args:
            feat (Tensor): Feature tensor of shape :math:`(N, E, H, W)`.
            img_metas (dict): Unused.

        Returns:
            Tensor: A tensor of shape :math:`(N, W-4, E)`.
        """
        x = self.convs(feat)  # (N, E, 1, W)
        return self.crnn(x, None)

# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import BaseModule

from mmocr.models.builder import BACKBONES
from mmocr.models.textrecog.backbones.resnet_abi import ResNetABI
from mmocr.models.textrecog.encoders.transformer_encoder import TFEncoder
from mmocr.models.textrecog.layers import PositionalEncoding


@BACKBONES.register_module()
class ResTransformer(BaseModule):
    """Implement ResTransformer backbone for text recognition, modified from
    `<https://github.com/FangShancheng/ABINet>`.

    Args:
        base_channels (int): Number of channels of input image tensor.
        layers (list[int]): List of BasicBlock number for each stage.
        channels (list[int]): List of out_channels of Conv2d layer.
        out_indices (None | Sequence[int]): Indices of output stages.
        last_stage_pool (bool): If True, add `MaxPool2d` layer to last stage.
    """

    def __init__(self,
                 n_layers=2,
                 n_head=8,
                 d_model=512,
                 d_inner=2048,
                 dropout=0.1,
                 max_len=8 * 32,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        assert d_model % n_head == 0, 'd_model must be divisible by n_head'

        self.resnet = ResNetABI()
        self.pos_encoder = PositionalEncoding(d_model, n_position=max_len)
        self.transformer = TFEncoder(
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_model // n_head,
            d_v=d_model // n_head,
            d_model=d_model,
            d_inner=d_inner,
            dropout=dropout)

    def forward(self, images):
        feature = self.resnet(images)
        n, c, h, w = feature.shape
        feature = feature.view(n, c, -1).permute(2, 0, 1)
        feature = self.pos_encoder(feature)
        feature = feature.permute(1, 2, 0).view(n, c, h, w)
        feature = self.transformer(feature)
        return feature

# Copyright (c) OpenMMLab. All rights reserved.
import copy

from mmcv.cnn.bricks.transformer import BaseTransformerLayer
from mmcv.runner import BaseModule, ModuleList

from mmocr.models.builder import BACKBONES
from mmocr.models.textrecog.backbones.resnet_abi import ResNetABI
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
        encoder_layer = BaseTransformerLayer(
            operation_order=('self_attn', 'norm', 'ffn', 'norm'),
            attn_cfgs=dict(
                type='MultiheadAttention',
                embed_dims=d_model,
                num_heads=n_head,
                attn_drop=dropout,
                dropout_layer=dict(type='Dropout', drop_prob=dropout),
            ),
            ffn_cfgs=dict(
                type='FFN',
                embed_dims=d_model,
                feedforward_channels=d_inner,
                ffn_drop=dropout,
            ),
            norm_cfg=dict(type='LN'),
        )
        self.transformer = ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(n_layers)])

    def forward(self, images):
        feature = self.resnet(images)
        n, c, h, w = feature.shape
        feature = feature.view(n, c, -1).transpose(1, 2)  # (n, h*w, c)
        feature = self.pos_encoder(feature)  # (n, h*w, c)
        feature = feature.transpose(0, 1)  # (h*w, n, c)
        for m in self.transformer:
            feature = m(feature)
        feature = feature.permute(1, 2, 0).view(n, c, h, w)
        return feature

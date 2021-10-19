# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from mmocr.models.builder import ENCODERS
from mmocr.models.textrecog.layers import PositionAttention
from .base_encoder import BaseEncoder


@ENCODERS.register_module()
class ABIVisionEncoder(BaseEncoder):
    """Converts visual features into text characters.

    Implementation of VisionEncoder in
        `ABINet <https://arxiv.org/abs/1910.04396>`_.

    Args:
        in_channels (int): Number of channels :math:`E` of input vector.
        num_channels (int): Number of channels of hidden vectors in mini U-Net.
        h (int): Height :math:`H` of input image features.
        w (int): Width :math:`W` of input image features.

        in_channels (int): Number of channels of input image features.
        num_channels (int): Number of channels of hidden vectors in mini U-Net.
        attn_height (int): Height :math:`H` of input image features.
        attn_width (int): Width :math:`W` of input image features.
        attn_mode (str): Upsampling mode for :obj:`torch.nn.Upsample` in mini
            U-Net.
        max_seq_len (int): Maximum text sequence length :math:`T`.
        num_chars (int): Number of text characters :math:`C`.
        init_cfg (dict): Specifies the initialization method for model layers.
    """

    def __init__(self,
                 in_channels=512,
                 num_channels=64,
                 attn_height=8,
                 attn_width=32,
                 attn_mode='nearest',
                 max_seq_len=40,
                 num_chars=90,
                 init_cfg=dict(type='Xavier', layer='Conv2d'),
                 **kwargs):
        super().__init__(init_cfg=init_cfg)

        self.attention = PositionAttention(
            max_length=max_seq_len,
            in_channels=in_channels,
            num_channels=num_channels,
            mode=attn_mode,
            h=attn_height,
            w=attn_width,
        )
        self.cls = nn.Linear(in_channels, num_chars)

    def forward(self, feat, img_metas=None):
        """
        Args:
            feat (Tensor): Image features of shape (N, E, H, W).

        Returns:
            A dict with keys ``feature``, ``logits`` and ``attn_scores``.
            feature (Tensor): Shape (N, T, E). Raw visual features for language
                decoder.
            logits (Tensor): Shape (N, T, C). The raw logits for characters.
            attn_scores (Tensor): Shape (N, T, H, W). Intermediate result for
                vision-language aligner.
        """
        attn_vecs, attn_scores = self.attention(feat)
        logits = self.cls(attn_vecs)
        result = {
            'feature': attn_vecs,
            'logits': logits,
            'attn_scores': attn_scores
        }

        return result

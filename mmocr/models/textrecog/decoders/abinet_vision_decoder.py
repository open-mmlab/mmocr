# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmocr.models.builder import DECODERS
from mmocr.models.common.modules import PositionalEncoding
from .base_decoder import BaseDecoder


@DECODERS.register_module()
class ABIVisionDecoder(BaseDecoder):
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

        self.max_seq_len = max_seq_len

        # For mini-Unet
        self.k_encoder = nn.Sequential(
            self._encoder_layer(in_channels, num_channels, stride=(1, 2)),
            self._encoder_layer(num_channels, num_channels, stride=(2, 2)),
            self._encoder_layer(num_channels, num_channels, stride=(2, 2)),
            self._encoder_layer(num_channels, num_channels, stride=(2, 2)))

        self.k_decoder = nn.Sequential(
            self._decoder_layer(
                num_channels, num_channels, scale_factor=2, mode=attn_mode),
            self._decoder_layer(
                num_channels, num_channels, scale_factor=2, mode=attn_mode),
            self._decoder_layer(
                num_channels, num_channels, scale_factor=2, mode=attn_mode),
            self._decoder_layer(
                num_channels,
                in_channels,
                size=(attn_height, attn_width),
                mode=attn_mode))

        self.pos_encoder = PositionalEncoding(in_channels, max_seq_len)
        self.project = nn.Linear(in_channels, in_channels)
        self.cls = nn.Linear(in_channels, num_chars)

    def forward_train(self,
                      feat,
                      out_enc=None,
                      targets_dict=None,
                      img_metas=None):
        """
        Args:
            feat (Tensor): Image features of shape (N, E, H, W).

        Returns:
            dict: A dict with keys ``feature``, ``logits`` and ``attn_scores``.

            - | feature (Tensor): Shape (N, T, E). Raw visual features for
                language decoder.
            - | logits (Tensor): Shape (N, T, C). The raw logits for
                characters.
            - | attn_scores (Tensor): Shape (N, T, H, W). Intermediate result
                for vision-language aligner.
        """
        # Position Attention
        N, E, H, W = feat.size()
        k, v = feat, feat  # (N, E, H, W)

        # Apply mini U-Net on k
        features = []
        for i in range(len(self.k_encoder)):
            k = self.k_encoder[i](k)
            features.append(k)
        for i in range(len(self.k_decoder) - 1):
            k = self.k_decoder[i](k)
            k = k + features[len(self.k_decoder) - 2 - i]
        k = self.k_decoder[-1](k)

        # q = positional encoding
        zeros = feat.new_zeros((N, self.max_seq_len, E))  # (N, T, E)
        q = self.pos_encoder(zeros)  # (N, T, E)
        q = self.project(q)  # (N, T, E)

        # Attention encoding
        attn_scores = torch.bmm(q, k.flatten(2, 3))  # (N, T, (H*W))
        attn_scores = attn_scores / (E**0.5)
        attn_scores = torch.softmax(attn_scores, dim=-1)
        v = v.permute(0, 2, 3, 1).view(N, -1, E)  # (N, (H*W), E)
        attn_vecs = torch.bmm(attn_scores, v)  # (N, T, E)

        logits = self.cls(attn_vecs)
        result = {
            'feature': attn_vecs,
            'logits': logits,
            'attn_scores': attn_scores.view(N, -1, H, W)
        }
        return result

    def forward_test(self, feat, out_enc=None, img_metas=None):
        return self.forward_train(feat, out_enc=out_enc, img_metas=img_metas)

    def _encoder_layer(self,
                       in_channels,
                       out_channels,
                       kernel_size=3,
                       stride=2,
                       padding=1):
        return ConvModule(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))

    def _decoder_layer(self,
                       in_channels,
                       out_channels,
                       kernel_size=3,
                       stride=1,
                       padding=1,
                       mode='nearest',
                       scale_factor=None,
                       size=None):
        align_corners = None if mode == 'nearest' else True
        return nn.Sequential(
            nn.Upsample(
                size=size,
                scale_factor=scale_factor,
                mode=mode,
                align_corners=align_corners),
            ConvModule(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU')))

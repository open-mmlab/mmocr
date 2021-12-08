# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.models.builder import ENCODERS, build_decoder, build_encoder
from .base_encoder import BaseEncoder


@ENCODERS.register_module()
class ABIVisionModel(BaseEncoder):
    """A wrapper of visual feature encoder and language token decoder that
    converts visual features into text tokens.

    Implementation of VisionEncoder in
        `ABINet <https://arxiv.org/abs/1910.04396>`_.

    Args:
        encoder (dict): Config for image feature encoder.
        decoder (dict): Config for language token decoder.
        init_cfg (dict): Specifies the initialization method for model layers.
    """

    def __init__(self,
                 encoder=dict(type='ResTransformer'),
                 decoder=dict(type='ABIVisionDecoder'),
                 init_cfg=dict(type='Xavier', layer='Conv2d'),
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.encoder = build_encoder(encoder)
        self.decoder = build_decoder(decoder)

    def forward(self, feat, img_metas=None):
        """
        Args:
            feat (Tensor): Images of shape (N, E, H, W).

        Returns:
            dict: A dict with keys ``feature``, ``logits`` and ``attn_scores``.

            - | feature (Tensor): Shape (N, T, E). Raw visual features for
                language decoder.
            - | logits (Tensor): Shape (N, T, C). The raw logits for
                characters. C is the number of characters.
            - | attn_scores (Tensor): Shape (N, T, H, W). Intermediate result
                for vision-language aligner.
        """
        feat = self.encoder(feat)
        return self.decoder(feat=feat, out_enc=None)

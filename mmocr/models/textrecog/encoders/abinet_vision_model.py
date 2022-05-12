# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.registry import MODELS
from .base_encoder import BaseEncoder


@MODELS.register_module()
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
                 encoder=dict(type='TransformerEncoder'),
                 decoder=dict(type='ABIVisionDecoder'),
                 init_cfg=dict(type='Xavier', layer='Conv2d'),
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.encoder = MODELS.build(encoder)
        self.decoder = MODELS.build(decoder)

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

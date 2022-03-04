# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmcv.runner.base_module import BaseModule

from mmocr.models.builder import build_decoder, build_encoder, build_loss


class BaseModality(BaseModule):
    """Base class for majormodality and extramodality."""

    def __init__(self,
                 encoder=None,
                 decoder=None,
                 loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):

        super().__init__(init_cfg=init_cfg)

        # Encoder module
        self.encoder = None
        if encoder is not None:
            self.encoder = build_encoder(encoder)

        # Decoder module
        if decoder is not None:
            self.decoder = build_decoder(decoder)

        # Loss
        assert loss is not None
        self.loss = build_loss(loss)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if pretrained is not None:
            warnings.warn('DeprecationWarning: pretrained is a deprecated \
                key, please consider using init_cfg')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

    def forward_train(self, feat, img_metas, targets_dict):
        """
        Args:
            feat (tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A list of image info dict where each dict
                contains: 'img_shape', 'filename', and may also contain
                'ori_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.

        Returns:
            dict[str, tensor]: A dictionary of loss components.
        """

        out_enc = None
        if self.encoder is not None:
            out_enc = self.encoder(feat, img_metas)

        out_dec = self.decoder(
            feat, out_enc, targets_dict, img_metas, train_mode=True)

        loss_inputs = (
            out_dec,
            targets_dict,
            img_metas,
        )
        losses = self.loss(*loss_inputs)

        return losses

    def simple_test(self, feat, img_metas, **kwargs):

        out_enc = None
        if self.encoder is not None:
            out_enc = self.encoder(feat, img_metas)

        out_dec = self.decoder(
            feat, out_enc, None, img_metas, train_mode=False)

        return out_dec

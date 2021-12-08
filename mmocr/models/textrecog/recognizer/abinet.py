# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from mmocr.models.builder import (DETECTORS, build_backbone, build_convertor,
                                  build_decoder, build_encoder, build_fuser,
                                  build_loss, build_preprocessor)
from .encode_decode_recognizer import EncodeDecodeRecognizer


@DETECTORS.register_module()
class ABINet(EncodeDecodeRecognizer):
    """Implementation of `Read Like Humans: Autonomous, Bidirectional and
    Iterative LanguageModeling for Scene Text Recognition.

    <https://arxiv.org/pdf/2103.06495.pdf>`_
    """

    def __init__(self,
                 preprocessor=None,
                 backbone=None,
                 encoder=None,
                 decoder=None,
                 iter_size=1,
                 fuser=None,
                 loss=None,
                 label_convertor=None,
                 train_cfg=None,
                 test_cfg=None,
                 max_seq_len=40,
                 pretrained=None,
                 init_cfg=None):
        super(EncodeDecodeRecognizer, self).__init__(init_cfg=init_cfg)

        # Label convertor (str2tensor, tensor2str)
        assert label_convertor is not None
        label_convertor.update(max_seq_len=max_seq_len)
        self.label_convertor = build_convertor(label_convertor)

        # Preprocessor module, e.g., TPS
        self.preprocessor = None
        if preprocessor is not None:
            self.preprocessor = build_preprocessor(preprocessor)

        # Backbone
        assert backbone is not None
        self.backbone = build_backbone(backbone)

        # Encoder module
        self.encoder = None
        if encoder is not None:
            self.encoder = build_encoder(encoder)

        # Decoder module
        self.decoder = None
        if decoder is not None:
            decoder.update(num_classes=self.label_convertor.num_classes())
            decoder.update(start_idx=self.label_convertor.start_idx)
            decoder.update(padding_idx=self.label_convertor.padding_idx)
            decoder.update(max_seq_len=max_seq_len)
            self.decoder = build_decoder(decoder)

        # Loss
        assert loss is not None
        self.loss = build_loss(loss)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.max_seq_len = max_seq_len

        if pretrained is not None:
            warnings.warn('DeprecationWarning: pretrained is a deprecated \
                key, please consider using init_cfg')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        self.iter_size = iter_size

        self.fuser = None
        if fuser is not None:
            self.fuser = build_fuser(fuser)

    def forward_train(self, img, img_metas):
        """
        Args:
            img (tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A list of image info dict where each dict
                contains: 'img_shape', 'filename', and may also contain
                'ori_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.

        Returns:
            dict[str, tensor]: A dictionary of loss components.
        """
        for img_meta in img_metas:
            valid_ratio = 1.0 * img_meta['resize_shape'][1] / img.size(-1)
            img_meta['valid_ratio'] = valid_ratio

        feat = self.extract_feat(img)

        gt_labels = [img_meta['text'] for img_meta in img_metas]

        targets_dict = self.label_convertor.str2tensor(gt_labels)

        text_logits = None
        out_enc = None
        if self.encoder is not None:
            out_enc = self.encoder(feat)
            text_logits = out_enc['logits']

        out_decs = []
        out_fusers = []
        for _ in range(self.iter_size):
            if self.decoder is not None:
                out_dec = self.decoder(
                    feat,
                    text_logits,
                    targets_dict,
                    img_metas,
                    train_mode=True)
                out_decs.append(out_dec)

            if self.fuser is not None:
                out_fuser = self.fuser(out_enc['feature'], out_dec['feature'])
                text_logits = out_fuser['logits']
                out_fusers.append(out_fuser)

        outputs = dict(
            out_enc=out_enc, out_decs=out_decs, out_fusers=out_fusers)

        losses = self.loss(outputs, targets_dict, img_metas)

        return losses

    def simple_test(self, img, img_metas, **kwargs):
        """Test function with test time augmentation.

        Args:
            imgs (torch.Tensor): Image input tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            list[str]: Text label result of each image.
        """
        for img_meta in img_metas:
            valid_ratio = 1.0 * img_meta['resize_shape'][1] / img.size(-1)
            img_meta['valid_ratio'] = valid_ratio

        feat = self.extract_feat(img)

        text_logits = None
        out_enc = None
        if self.encoder is not None:
            out_enc = self.encoder(feat)
            text_logits = out_enc['logits']

        out_decs = []
        out_fusers = []
        for _ in range(self.iter_size):
            if self.decoder is not None:
                out_dec = self.decoder(
                    feat, text_logits, img_metas=img_metas, train_mode=False)
                out_decs.append(out_dec)

            if self.fuser is not None:
                out_fuser = self.fuser(out_enc['feature'], out_dec['feature'])
                text_logits = out_fuser['logits']
                out_fusers.append(out_fuser)

        if len(out_fusers) > 0:
            ret = out_fusers[-1]
        elif len(out_decs) > 0:
            ret = out_decs[-1]
        else:
            ret = out_enc

        # early return to avoid post processing
        if torch.onnx.is_in_onnx_export():
            return ret['logits']

        label_indexes, label_scores = self.label_convertor.tensor2idx(
            ret['logits'], img_metas)
        label_strings = self.label_convertor.idx2str(label_indexes)

        # flatten batch results
        results = []
        for string, score in zip(label_strings, label_scores):
            results.append(dict(text=string, score=score))

        return results

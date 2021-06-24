import warnings

import torch
from mmdet.models.builder import DETECTORS, build_backbone, build_loss

from mmocr.models.builder import (build_convertor, build_decoder,
                                  build_encoder, build_preprocessor)
from .base import BaseRecognizer


@DETECTORS.register_module()
class EncodeDecodeRecognizer(BaseRecognizer):
    """Base class for encode-decode recognizer."""

    def __init__(self,
                 preprocessor=None,
                 backbone=None,
                 encoder=None,
                 decoder=None,
                 loss=None,
                 label_convertor=None,
                 train_cfg=None,
                 test_cfg=None,
                 max_seq_len=40,
                 pretrained=None,
                 init_cfg=None):
        if pretrained is not None:
            warnings.warn('DeprecationWarning: pretrained is a deprecated \
                key, please consider using init_cfg')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        super().__init__(init_cfg=init_cfg)

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
        assert decoder is not None
        decoder.update(num_classes=self.label_convertor.num_classes())
        decoder.update(start_idx=self.label_convertor.start_idx)
        decoder.update(padding_idx=self.label_convertor.padding_idx)
        decoder.update(max_seq_len=max_seq_len)
        self.decoder = build_decoder(decoder)

        # Loss
        assert loss is not None
        loss.update(ignore_index=self.label_convertor.padding_idx)
        self.loss = build_loss(loss)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.max_seq_len = max_seq_len

        # self.init_weights(pretrained=pretrained)

    '''
    def init_weights(self, pretrained=None):
        """Initialize the weights of recognizer."""
        super().init_weights(pretrained)

        if self.preprocessor is not None:
            self.preprocessor.init_weights()

        self.backbone.init_weights()

        if self.encoder is not None:
            self.encoder.init_weights()

        self.decoder.init_weights()
    '''

    def extract_feat(self, img):
        """Directly extract features from the backbone."""
        if self.preprocessor is not None:
            img = self.preprocessor(img)

        x = self.backbone(img)

        return x

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
        feat = self.extract_feat(img)

        gt_labels = [img_meta['text'] for img_meta in img_metas]

        targets_dict = self.label_convertor.str2tensor(gt_labels)

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

    def simple_test(self, img, img_metas, **kwargs):
        """Test function with test time augmentation.

        Args:
            imgs (torch.Tensor): Image input tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            list[str]: Text label result of each image.
        """
        feat = self.extract_feat(img)

        out_enc = None
        if self.encoder is not None:
            out_enc = self.encoder(feat, img_metas)

        out_dec = self.decoder(
            feat, out_enc, None, img_metas, train_mode=False)

        # early return to avoid post processing
        if torch.onnx.is_in_onnx_export():
            return out_dec

        label_indexes, label_scores = self.label_convertor.tensor2idx(
            out_dec, img_metas)
        label_strings = self.label_convertor.idx2str(label_indexes)

        # flatten batch results
        results = []
        for string, score in zip(label_strings, label_scores):
            results.append(dict(text=string, score=score))

        return results

    def merge_aug_results(self, aug_results):
        out_text, out_score = '', -1
        for result in aug_results:
            text = result[0]['text']
            score = sum(result[0]['score']) / max(1, len(text))
            if score > out_score:
                out_text = text
                out_score = score
        out_results = [dict(text=out_text, score=out_score)]
        return out_results

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function as well as time augmentation.

        Args:
            imgs (list[tensor]): Tensor should have shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): The metadata of images.
        """
        aug_results = []
        for img, img_meta in zip(imgs, img_metas):
            result = self.simple_test(img, img_meta, **kwargs)
            aug_results.append(result)

        return self.merge_aug_results(aug_results)

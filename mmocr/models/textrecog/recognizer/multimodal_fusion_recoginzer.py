# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from mmocr.models.builder import (RECOGNIZERS, build_backbone, build_convertor,
                                  build_extra_modality, build_fusion,
                                  build_major_modality, build_preprocessor)
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class MultiModalFusionRecognizer(BaseRecognizer):
    """"""

    def __init__(self,
                 preprocessor=None,
                 backbone=None,
                 major_modality=None,
                 extra_modality=None,
                 fusion=None,
                 label_convertor=None,
                 train_cfg=None,
                 test_cfg=None,
                 max_seq_len=40,
                 pretrained=None,
                 init_cfg=None):

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
        if backbone:
            self.backbone = build_backbone(backbone)

        # base modality module
        if major_modality:
            self.major_modality = build_major_modality(major_modality,
                                                       train_cfg, test_cfg)

        # extra modality module
        if extra_modality:
            self.extra_modality = build_extra_modality(extra_modality,
                                                       train_cfg, test_cfg)

        if fusion:
            self.fusion = build_fusion(fusion, train_cfg, test_cfg)
        # Loss

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.max_seq_len = max_seq_len

        if pretrained is not None:
            warnings.warn('DeprecationWarning: pretrained is a deprecated \
                key, please consider using init_cfg')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

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
        for img_meta in img_metas:
            valid_ratio = 1.0 * img_meta['resize_shape'][1] / img.size(-1)
            img_meta['valid_ratio'] = valid_ratio

        feat = self.extract_feat(img)

        gt_labels = [img_meta['text'] for img_meta in img_metas]

        targets_dict = self.label_convertor.str2tensor(gt_labels)

        out_modality = None
        losses = dict()
        if self.major_modality is not None:
            major_modality_loss, out_modality = self.major_modality(
                feat, out_modality, targets_dict, img_metas, train_mode=True)
            losses.update(major_modality_loss)

        if self.extra_modality is not None:
            extra_modality_loss, out_modality = self.extra_modality(
                feat, out_modality, targets_dict, img_metas, train_mode=True)
            losses.update(extra_modality_loss)

        if self.fusion is not None:
            fusion_loss, out_modality = self.fusion(
                feat, out_modality, targets_dict, img_metas, train_mode=True)
            losses.update(fusion_loss)
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

        out_modality = None
        if self.major_modality is not None:
            out_modality = self.major_modality(
                feat, img_metas, train_mode=False)

        if self.extra_modality is not None:
            out_modality = self.extra_modality(
                feat, out_modality, img_metas, train_mode=False)

        if self.fusion is not None:
            out_modality = self.fusion(
                feat, out_modality, img_metas, train_mode=False)

        # early return to avoid post processing
        if torch.onnx.is_in_onnx_export():
            return out_modality

        label_indexes, label_scores = self.label_convertor.tensor2idx(
            out_modality, img_metas)
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

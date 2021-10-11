# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmocr.models.builder import DETECTORS, build_fuser
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
        super().__init__(
            preprocessor=preprocessor,
            backbone=backbone,
            encoder=encoder,
            decoder=decoder,
            loss=loss,
            label_convertor=label_convertor,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            max_seq_len=max_seq_len,
            pretrained=pretrained,
            init_cfg=init_cfg)

        assert encoder is not None

        self.iter_size = iter_size

        assert fuser is not None
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

        out_enc = self.encoder(feat, img_metas)

        text_logits = out_enc['logits']
        out_decs = []
        out_fusers = []
        for _ in range(self.iter_size):
            out_dec = self.decoder(
                feat, text_logits, targets_dict, img_metas, train_mode=True)

            out_fuser = self.fuser(out_enc['feature'], out_dec['feature'])
            text_logits = out_fuser['logits']
            out_decs.append(out_dec)
            out_fusers.append(out_fuser)

        label_indexes, label_scores = self.label_convertor.tensor2idx(
            out_decs[-1]['logits'], img_metas)
        label_strings = self.label_convertor.idx2str(label_indexes)
        print(label_strings)

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

        out_enc = self.encoder(feat, img_metas)

        text_logits = out_enc['logits']
        out_decs = []
        out_fusers = []
        for _ in range(self.iter_size):
            out_dec = self.decoder(
                feat, text_logits, img_metas=img_metas, train_mode=False)

            out_fuser = self.fuser(out_enc['feature'], out_dec['feature'])
            text_logits = out_fuser['logits']
            out_decs.append(out_dec)
            out_fusers.append(out_fuser)

        # early return to avoid post processing
        if torch.onnx.is_in_onnx_export():
            return out_fusers[-1]['logits']

        label_indexes, label_scores = self.label_convertor.tensor2idx(
            out_decs[-1]['logits'], img_metas)
        label_strings = self.label_convertor.idx2str(label_indexes)
        print(label_strings)

        # flatten batch results
        results = []
        for string, score in zip(label_strings, label_scores):
            results.append(dict(text=string, score=score))

        return results

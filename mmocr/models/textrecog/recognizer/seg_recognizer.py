import warnings

from mmocr.models.builder import (DETECTORS, build_backbone, build_convertor,
                                  build_head, build_loss, build_neck,
                                  build_preprocessor)
from .base import BaseRecognizer


@DETECTORS.register_module()
class SegRecognizer(BaseRecognizer):
    """Base class for segmentation based recognizer."""

    def __init__(self,
                 preprocessor=None,
                 backbone=None,
                 neck=None,
                 head=None,
                 loss=None,
                 label_convertor=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        # Label_convertor
        assert label_convertor is not None
        self.label_convertor = build_convertor(label_convertor)

        # Preprocessor module, e.g., TPS
        self.preprocessor = None
        if preprocessor is not None:
            self.preprocessor = build_preprocessor(preprocessor)

        # Backbone
        assert backbone is not None
        self.backbone = build_backbone(backbone)

        # Neck
        assert neck is not None
        self.neck = build_neck(neck)

        # Head
        assert head is not None
        head.update(num_classes=self.label_convertor.num_classes())
        self.head = build_head(head)

        # Loss
        assert loss is not None
        self.loss = build_loss(loss)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
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

    def forward_train(self, img, img_metas, gt_kernels=None):
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

        feats = self.extract_feat(img)

        out_neck = self.neck(feats)

        out_head = self.head(out_neck)

        loss_inputs = (out_neck, out_head, gt_kernels)

        losses = self.loss(*loss_inputs)

        return losses

    def simple_test(self, img, img_metas, **kwargs):
        """Test function without test time augmentation.

        Args:
            imgs (torch.Tensor): Image input tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            list[str]: Text label result of each image.
        """

        feat = self.extract_feat(img)

        out_neck = self.neck(feat)

        out_head = self.head(out_neck)

        for img_meta in img_metas:
            valid_ratio = 1.0 * img_meta['resize_shape'][1] / img.size(-1)
            img_meta['valid_ratio'] = valid_ratio

        texts, scores = self.label_convertor.tensor2str(out_head, img_metas)

        # flatten batch results
        results = []
        for text, score in zip(texts, scores):
            results.append(dict(text=text, score=score))

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
        """Test function with test time augmentation.

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

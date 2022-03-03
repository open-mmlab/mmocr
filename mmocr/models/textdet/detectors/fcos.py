# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.models.builder import DETECTORS, build_loss, build_postprocessor
from .single_stage_text_detector import SingleStageTextDetector
from .text_detector_mixin import TextDetectorMixin


@DETECTORS.register_module()
class FCOS(TextDetectorMixin, SingleStageTextDetector):
    """The class for implementing ABCNet text detector: ABCNet: Real-time Scene
    Text Spotting with Adaptive Bezier-Curve Network.

    [https://arxiv.org/abs/2002.10200].
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 loss,
                 postprocessor,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 show_score=False,
                 init_cfg=None):
        SingleStageTextDetector.__init__(self, backbone, neck, bbox_head,
                                         train_cfg, test_cfg, pretrained,
                                         init_cfg)
        TextDetectorMixin.__init__(self, show_score)
        postprocessor.update(train_cfg=train_cfg)
        postprocessor.update(test_cfg=test_cfg)
        self.postprocessor = build_postprocessor(postprocessor)
        self.loss = build_loss(loss)

    def forward_train(self, img, img_metas, **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys, see
                :class:`mmdet.datasets.pipelines.Collect`.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.extract_feat(img)
        preds = self.bbox_head(x)
        losses = self.loss(preds, img_metas, **kwargs)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        outputs = self.postprocessor(outs, img_metas)
        return outputs

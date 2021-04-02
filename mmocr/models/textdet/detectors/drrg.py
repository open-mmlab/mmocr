from mmdet.models.builder import DETECTORS
from . import SingleStageTextDetector, TextDetectorMixin


@DETECTORS.register_module()
class DRRG(TextDetectorMixin, SingleStageTextDetector):
    """The class for implementing DRRG text detector: Deep Relational Reasoning
    Graph Network for Arbitrary Shape Text Detection.

    [https://arxiv.org/abs/2003.07493]
    """

    def forward_train(self, img, img_metas, **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details of the values of these keys, see
                :class:`mmdet.datasets.pipelines.Collect`.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.extract_feat(img)
        gt_comp_attribs = kwargs.pop('gt_comp_attribs')
        preds = self.bbox_head(x, gt_comp_attribs)
        losses = self.bbox_head.loss(preds, **kwargs)
        return losses

    def simple_test(self, img, img_metas, rescale=False):

        x = self.extract_feat(img)
        outs = self.bbox_head.single_test(x, img)
        boundaries = self.bbox_head.get_boundary(*outs, img_metas, rescale)

        return [boundaries]

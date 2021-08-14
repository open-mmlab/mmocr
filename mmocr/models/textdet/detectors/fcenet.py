from mmocr.models.builder import DETECTORS
from .single_stage_text_detector import SingleStageTextDetector
from .text_detector_mixin import TextDetectorMixin


@DETECTORS.register_module()
class FCENet(TextDetectorMixin, SingleStageTextDetector):
    """The class for implementing FCENet text detector
    FCENet(CVPR2021): Fourier Contour Embedding for Arbitrary-shaped Text
        Detection

    [https://arxiv.org/abs/2104.10442]
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 show_score=False,
                 init_cfg=None):
        SingleStageTextDetector.__init__(self, backbone, neck, bbox_head,
                                         train_cfg, test_cfg, pretrained,
                                         init_cfg)
        TextDetectorMixin.__init__(self, show_score)

    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        boundaries = self.bbox_head.get_boundary(outs, img_metas, rescale)

        return [boundaries]

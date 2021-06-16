from mmdet.models.builder import DETECTORS

from mmocr.models.textdet.detectors.single_stage_text_detector import \
    SingleStageTextDetector
from mmocr.models.textdet.detectors.text_detector_mixin import \
    TextDetectorMixin


@DETECTORS.register_module()
class PANet(TextDetectorMixin, SingleStageTextDetector):
    """The class for implementing PANet text detector:

    Efficient and Accurate Arbitrary-Shaped Text Detection with Pixel
    Aggregation Network [https://arxiv.org/abs/1908.05900].
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 show_score=False):
        SingleStageTextDetector.__init__(self, backbone, neck, bbox_head,
                                         train_cfg, test_cfg, pretrained)
        TextDetectorMixin.__init__(self, show_score)

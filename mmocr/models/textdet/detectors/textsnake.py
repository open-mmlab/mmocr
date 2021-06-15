from mmdet.models.builder import DETECTORS

from . import SingleStageTextDetector, TextDetectorMixin


@DETECTORS.register_module()
class TextSnake(TextDetectorMixin, SingleStageTextDetector):
    """The class for implementing TextSnake text detector: TextSnake: A
    Flexible Representation for Detecting Text of Arbitrary Shapes.

    [https://arxiv.org/abs/1807.01544]
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

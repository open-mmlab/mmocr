from mmdet.models.builder import DETECTORS
from .seg_recognizer import SegRecognizer


@DETECTORS.register_module()
class CAFCNNet(SegRecognizer):
    """Implementation of `CAFCN <https://arxiv.org/pdf/1809.06508.pdf>`_"""

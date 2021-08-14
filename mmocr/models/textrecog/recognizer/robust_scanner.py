from mmocr.models.builder import DETECTORS
from .encode_decode_recognizer import EncodeDecodeRecognizer


@DETECTORS.register_module()
class RobustScanner(EncodeDecodeRecognizer):
    """Implementation of `RobustScanner.

    <https://arxiv.org/pdf/2007.07542.pdf>
    """

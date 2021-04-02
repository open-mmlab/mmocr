from mmdet.models.builder import DETECTORS
from .encode_decode_recognizer import EncodeDecodeRecognizer


@DETECTORS.register_module()
class TransformerNet(EncodeDecodeRecognizer):
    """Implementation of Transformer based OCR."""

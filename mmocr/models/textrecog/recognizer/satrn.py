from mmocr.models.builder import DETECTORS
from .encode_decode_recognizer import EncodeDecodeRecognizer


@DETECTORS.register_module()
class SATRN(EncodeDecodeRecognizer):
    """Implementation of `SATRN <https://arxiv.org/abs/1910.04396>`_"""

import torch
import torch.nn.functional as F

from mmdet.models.builder import DETECTORS
from .encode_decode_recognizer import EncodeDecodeRecognizer


@DETECTORS.register_module()
class CRNNNet(EncodeDecodeRecognizer):
    """CTC-loss based recognizer."""

    def forward_conversion(self, params, img):
        x = self.extract_feat(img)
        x = self.encoder(x)
        outs = self.decoder(x)
        outs = F.softmax(outs, dim=2)
        params = torch.pow(params, 1)
        return outs, params

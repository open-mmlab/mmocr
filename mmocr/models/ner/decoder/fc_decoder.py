import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import uniform_init, xavier_init

from mmocr.models.builder import DECODERS


@DECODERS.register_module()
class FCDecoder(nn.Module):
    """FC Decoder class for Ner.

    Args:
        num_labels (int):
    """

    def __init__(self,
                 num_labels=None,
                 hidden_dropout_prob=0.1,
                 hidden_size=768):
        super().__init__()
        self.num_labels = num_labels

        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, self.num_labels)
        self.init_weights()

    def forward(self, outputs):
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        softmax = F.softmax(logits, dim=2)
        preds = softmax.detach().cpu().numpy()
        preds = np.argmax(preds, axis=2).tolist()
        return logits, preds

    def init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                uniform_init(m)

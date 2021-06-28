import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule

from mmocr.models.builder import DECODERS


@DECODERS.register_module()
class FCDecoder(BaseModule):
    """FC Decoder class for Ner.

    Args:
        num_labels (int): Number of categories mapped by entity label.
        hidden_dropout_prob (float): The dropout probability of hidden layer.
        hidden_size (int): Hidden layer output layer channels.
    """

    def __init__(self,
                 num_labels=None,
                 hidden_dropout_prob=0.1,
                 hidden_size=768,
                 init_cfg=[
                     dict(type='Xavier', layer='Conv2d'),
                     dict(type='Uniform', layer='BatchNorm2d')
                 ]):
        super().__init__(init_cfg=init_cfg)
        self.num_labels = num_labels

        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, self.num_labels)
        # self.init_weights()

    def forward(self, outputs):
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        softmax = F.softmax(logits, dim=2)
        preds = softmax.detach().cpu().numpy()
        preds = np.argmax(preds, axis=2).tolist()
        return logits, preds

    '''
    def init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                uniform_init(m)
    '''

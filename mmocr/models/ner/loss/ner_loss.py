from torch import nn
from torch.nn import CrossEntropyLoss

from mmdet.models.builder import LOSSES
from .focal_loss import FocalLoss
from .label_smoothing import LabelSmoothingCrossEntropy


@LOSSES.register_module()
class NerLoss(nn.Module):
    """"""

    def __init__(self, num_labels=None, loss_type=None, **kwargs):
        super(NerLoss, self).__init__()
        self.num_labels = num_labels
        self.loss_type = loss_type

    def forward(self, logits, labels, attention_mask):
        '''
        input: [N, C]
        target: [N, ]
        '''
        if labels is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy(ignore_index=0)
            elif self.loss_type == 'focal':

                loss_fct = FocalLoss(ignore_index=0)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            return loss

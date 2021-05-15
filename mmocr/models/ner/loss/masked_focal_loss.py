from torch import nn

from mmdet.models.builder import LOSSES
from mmocr.models.common.losses.focal_loss import FocalLoss


@LOSSES.register_module()
class MaskedFocalLoss(nn.Module):
    """The implementation of masked focal loss."""

    def __init__(self, num_labels=None, **kwargs):
        super().__init__()
        self.num_labels = num_labels
        self.criterion = FocalLoss(ignore_index=0)

    def forward(self, logits, img_metas):
        '''Loss forword.
        Args:
            logits: Model output with shape [N, C].
            img_metas (dict): A dict containing the following keys:
                    - img (list]): This parameter is reserved.
                    - labels (list[int]): []*max_len
                    - texts (list): []*max_len
                    - input_ids (list): []*max_len
                    - attention_mask (list): []*max_len
                    - token_type_ids (list): []*max_len
        '''

        labels = img_metas['labels']
        attention_masks = img_metas['attention_masks']

        # Only keep active parts of the loss
        if attention_masks is not None:
            active_loss = attention_masks.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = self.criterion(active_logits, active_labels)
        else:
            loss = self.criterion(
                logits.view(-1, self.num_labels), labels.view(-1))
        return {'loss_cls': loss}

from transformers.trainer_pt_utils import LabelSmoother

from mmocr.registry import MODELS


@MODELS.register_module()
class ComputeLossAfterLabelSmooth(LabelSmoother):
    """Compute loss after label-smoothing.

    Args:
        epsilon (`float`, *optional*, defaults to 0.1):
            The label smoothing factor.
        ignore_index (`int`, *optional*, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    def __call__(self, model_output, labels, shift_labels=False):
        loss = super().__call__(model_output, labels, shift_labels)
        return {'loss': loss}

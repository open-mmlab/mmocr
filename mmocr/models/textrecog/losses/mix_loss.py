# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmocr.models.builder import LOSSES


@LOSSES.register_module()
class ABILoss(nn.Module):
    """Implementation of ABINet multiloss that allows mixing different types of
    losses with weights.

    Args:
        enc_weight (float): The weight of encoder loss. Defaults to 1.0.
        dec_weight (float): The weight of decoder loss. Defaults to 1.0.
        fusion_weight (float): The weight of fuser (aligner) loss.
            Defaults to 1.0.
        num_classes (int): Number of unique output language tokens.

    Returns:
        A dictionary whose key/value pairs are the losses of three modules.
    """

    def __init__(self,
                 enc_weight=1.0,
                 dec_weight=1.0,
                 fusion_weight=1.0,
                 num_classes=37,
                 **kwargs):
        assert isinstance(enc_weight, float) or isinstance(enc_weight, int)
        assert isinstance(dec_weight, float) or isinstance(dec_weight, int)
        assert isinstance(fusion_weight, float) or \
            isinstance(fusion_weight, int)
        super().__init__()
        self.enc_weight = enc_weight
        self.dec_weight = dec_weight
        self.fusion_weight = fusion_weight
        self.num_classes = num_classes

    def _flatten(self, logits, target_lens):
        flatten_logits = torch.cat(
            [s[:target_lens[i]] for i, s in enumerate((logits))])
        return flatten_logits

    def _ce_loss(self, logits, targets):
        targets_one_hot = F.one_hot(targets, self.num_classes)
        log_prob = F.log_softmax(logits, dim=-1)
        loss = -(targets_one_hot.to(log_prob.device) * log_prob).sum(dim=-1)
        return loss.mean()

    def _loss_over_iters(self, outputs, targets):
        """
        Args:
            outputs (list[Tensor]): Each tensor has shape (N, T, C) where N is
                the batch size, T is the sequence length and C is the number of
                classes.
            targets_dicts (dict): The dictionary with at least `padded_targets`
                defined.
        """
        iter_num = len(outputs)
        dec_outputs = torch.cat(outputs, dim=0)
        flatten_targets_iternum = targets.repeat(iter_num)
        return self._ce_loss(dec_outputs, flatten_targets_iternum)

    def forward(self, outputs, targets_dict, img_metas=None):
        """
        Args:
            outputs (dict): The output dictionary with at least one of
                ``out_enc``, ``out_dec`` and ``out_fusers`` specified.
            targets_dict (dict): The target dictionary containing the key
                ``padded_targets``, which represents target sequences in
                shape (batch_size, sequence_length).

        Returns:
            A loss dictionary with ``loss_visual``, ``loss_lang`` and
            ``loss_fusion``. Each should either be the loss tensor or ``0`` if
            the output of its corresponding module is not given.
        """
        assert 'out_enc' in outputs or \
            'out_dec' in outputs or 'out_fusers' in outputs
        losses = {}

        target_lens = [len(t) for t in targets_dict['targets']]
        flatten_targets = torch.cat([t for t in targets_dict['targets']])

        if outputs.get('out_enc', None):
            enc_input = self._flatten(outputs['out_enc']['logits'],
                                      target_lens)
            enc_loss = self._ce_loss(enc_input,
                                     flatten_targets) * self.enc_weight
            losses['loss_visual'] = enc_loss
        if outputs.get('out_decs', None):
            dec_logits = [
                self._flatten(o['logits'], target_lens)
                for o in outputs['out_decs']
            ]
            dec_loss = self._loss_over_iters(dec_logits,
                                             flatten_targets) * self.dec_weight
            losses['loss_lang'] = dec_loss
        if outputs.get('out_fusers', None):
            fusion_logits = [
                self._flatten(o['logits'], target_lens)
                for o in outputs['out_fusers']
            ]
            fusion_loss = self._loss_over_iters(
                fusion_logits, flatten_targets) * self.fusion_weight
            losses['loss_fusion'] = fusion_loss
        return losses

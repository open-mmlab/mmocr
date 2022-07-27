# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn

from mmocr.structures import TextRecogDataSample
from mmocr.models.textrecog.dictionary import Dictionary
from mmocr.registry import MODELS
from .base_decoder import BaseDecoder


@MODELS.register_module()
class ABIFuser(BaseDecoder):
    r"""Transformer-based language model responsible for spell correction.
    Implementation of language model of \
        `ABINet <https://arxiv.org/abs/1910.04396>`_.

    Args:
        dictionary (dict or :obj:`Dictionary`): The config for `Dictionary` or
            the instance of `Dictionary`. The dictionary must have an end
            token.
        vision_decoder (dict): The config for vision decoder.
        language_decoder (dict, optional): The config for language decoder.
        num_iters (int): Rounds of iterative correction. Defaults to 1.
        d_model (int): Hidden size :math:`E` of model. Defaults to 512.
        max_seq_len (int): Maximum sequence length :math:`T`. The
            sequence is usually generated from decoder. Defaults to 40.
        module_loss (dict, optional): Config to build loss. Defaults to None.
        postprocessor (dict, optional): Config to build postprocessor.
            Defaults to None.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """

    def __init__(self,
                 dictionary: Union[Dict, Dictionary],
                 vision_decoder: Dict,
                 language_decoder: Optional[Dict] = None,
                 d_model: int = 512,
                 num_iters: int = 1,
                 max_seq_len: int = 40,
                 module_loss: Optional[Dict] = None,
                 postprocessor: Optional[Dict] = None,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None,
                 **kwargs) -> None:
        super().__init__(
            dictionary=dictionary,
            module_loss=module_loss,
            postprocessor=postprocessor,
            max_seq_len=max_seq_len,
            init_cfg=init_cfg)

        assert self.dictionary.end_idx is not None,\
            'Dictionary must contain an end token! (with_end=True)'

        self.d_model = d_model
        self.num_iters = num_iters
        if language_decoder is not None:
            self.w_att = nn.Linear(2 * d_model, d_model)
            self.cls = nn.Linear(d_model, self.dictionary.num_classes)

        self.vision_decoder = vision_decoder
        self.language_decoder = language_decoder
        for cfg_name in ['vision_decoder', 'language_decoder']:
            if getattr(self, cfg_name, None) is not None:
                cfg = getattr(self, cfg_name)
                if cfg.get('dictionary', None) is None:
                    cfg.update(dictionary=self.dictionary)
                else:
                    warnings.warn(f"Using dictionary {cfg['dictionary']} "
                                  "in decoder's config.")
                if cfg.get('max_seq_len', None) is None:
                    cfg.update(max_seq_len=max_seq_len)
                else:
                    warnings.warn(f"Using max_seq_len {cfg['max_seq_len']} "
                                  "in decoder's config.")
                setattr(self, cfg_name, MODELS.build(cfg))
        self.softmax = nn.Softmax(dim=-1)

    def forward_train(
            self,
            feat: Optional[torch.Tensor] = None,
            out_enc: torch.Tensor = None,
            data_samples: Optional[Sequence[TextRecogDataSample]] = None
    ) -> Dict:
        """
        Args:
            feat (torch.Tensor, optional): Not required. Feature map
                placeholder. Defaults to None.
            out_enc (Tensor): Raw language logitis. Shape :math:`(N, T, C)`.
                Defaults to None.
            data_samples (list[TextRecogDataSample], optional): Not required.
                DataSample placeholder. Defaults to None.

        Returns:
            A dict with keys ``out_enc``, ``out_decs`` and ``out_fusers``.

            - out_vis (dict):  Dict from ``self.vision_decoder`` with keys
              ``feature``, ``logits`` and ``attn_scores``.
            - out_langs (dict or list): Dict from ``self.vision_decoder`` with
              keys ``feature``, ``logits`` if applicable, or an empty list
              otherwise.
            - out_fusers (dict or list): Dict of fused visual and language
              features with keys ``feature``, ``logits`` if applicable, or
              an empty list otherwise.
        """
        out_vis = self.vision_decoder(feat, out_enc, data_samples)
        out_langs = []
        out_fusers = []
        if self.language_decoder is not None:
            text_logits = out_vis['logits']
            for _ in range(self.num_iters):
                out_dec = self.language_decoder(feat, text_logits,
                                                data_samples)
                out_langs.append(out_dec)
                out_fuser = self.fuse(out_vis['feature'], out_dec['feature'])
                text_logits = out_fuser['logits']
                out_fusers.append(out_fuser)

        outputs = dict(
            out_vis=out_vis, out_langs=out_langs, out_fusers=out_fusers)

        return outputs

    def forward_test(
        self,
        feat: Optional[torch.Tensor],
        logits: torch.Tensor,
        data_samples: Optional[Sequence[TextRecogDataSample]] = None
    ) -> torch.Tensor:
        """
        Args:
            feat (torch.Tensor, optional): Not required. Feature map
                placeholder. Defaults to None.
            logits (Tensor): Raw language logitis. Shape :math:`(N, T, C)`.
            data_samples (list[TextRecogDataSample], optional): Not required.
                DataSample placeholder. Defaults to None.

        Returns:
            Tensor: Character probabilities. of shape
            :math:`(N, self.max_seq_len, C)` where :math:`C` is
            ``num_classes``.
        """
        raw_result = self.forward_train(feat, logits, data_samples)

        if 'out_fusers' in raw_result and len(raw_result['out_fusers']) > 0:
            ret = raw_result['out_fusers'][-1]['logits']
        elif 'out_langs' in raw_result and len(raw_result['out_langs']) > 0:
            ret = raw_result['out_langs'][-1]['logits']
        else:
            ret = raw_result['out_vis']['logits']

        return self.softmax(ret)

    def fuse(self, l_feature: torch.Tensor, v_feature: torch.Tensor) -> Dict:
        """Mix and align visual feature and linguistic feature.

        Args:
            l_feature (torch.Tensor): (N, T, E) where T is length, N is batch
                size and E is dim of model.
            v_feature (torch.Tensor): (N, T, E) shape the same as l_feature.

        Returns:
            dict: A dict with key ``logits``. of shape :math:`(N, T, C)` where
            N is batch size, T is length and C is the number of characters.
        """
        f = torch.cat((l_feature, v_feature), dim=2)
        f_att = torch.sigmoid(self.w_att(f))
        output = f_att * v_feature + (1 - f_att) * l_feature

        logits = self.cls(output)  # (N, T, C)

        return {'logits': logits}

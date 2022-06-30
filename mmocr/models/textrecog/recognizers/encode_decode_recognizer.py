# Copyright (c) OpenMMLab. All rights reserved.

import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from mmocr.core.data_structures import TextRecogDataSample
from mmocr.registry import MODELS, TASK_UTILS
from .base import BaseRecognizer

ForwardResults = Union[Dict[str, torch.Tensor], List[TextRecogDataSample],
                       Tuple[torch.Tensor], torch.Tensor]
SampleList = List[TextRecogDataSample]
OptSampleList = Optional[SampleList]


@MODELS.register_module()
class EncodeDecodeRecognizer(BaseRecognizer):
    """Base class for encode-decode recognizer.

    Args:
        preprocessor (dict, optional): Config dict for preprocessor. Defaults
            to None.
        backbone (dict, optional): Backbone config. Defaults to None.
        encoder (dict, optional): Encoder config. If None, the output from
            backbone will be directly fed into ``decoder``. Defaults to None.
        decoder (dict, optional): Decoder config. Defaults to None.
        dictionary (dict, optional): Dictionary config. Defaults to None.
        max_seq_len (int): Maximum sequence length. Defaults to 40.
        data_preprocessor (dict, optional): Model preprocessing config
            for processing the input image data. Keys allowed are
            ``to_rgb``(bool), ``pad_size_divisor``(int), ``pad_value``(int or
            float), ``mean``(int or float) and ``std``(int or float).
            Preprcessing order: 1. to rgb; 2. normalization 3. pad.
            Defaults to None.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """

    def __init__(self,
                 preprocessor: Optional[Dict] = None,
                 backbone: Optional[Dict] = None,
                 encoder: Optional[Dict] = None,
                 decoder: Optional[Dict] = None,
                 dictionary: Optional[Dict] = None,
                 max_seq_len: int = 40,
                 data_preprocessor: Dict = None,
                 init_cfg: Optional[Dict] = None) -> None:

        super().__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        # Preprocessor module, e.g., TPS
        if preprocessor is not None:
            self.preprocessor = MODELS.build(preprocessor)

        # Backbone
        if backbone is not None:
            self.backbone = MODELS.build(backbone)

        # Encoder module
        if encoder is not None:
            self.encoder = MODELS.build(encoder)

        # Dictionary
        if dictionary is not None:
            self.dictionary = TASK_UTILS.build(dictionary)
        # Decoder module
        assert decoder is not None

        if self.with_dictionary:
            if decoder.get('dictionary', None) is None:
                decoder.update(dictionary=self.dictionary)
            else:
                warnings.warn(f"Using dictionary {decoder['dictionary']} "
                              "in decoder's config.")
        if decoder.get('max_seq_len', None) is None:
            decoder.update(max_seq_len=max_seq_len)
        else:
            warnings.warn(f"Using max_seq_len {decoder['max_seq_len']} "
                          "in decoder's config.")
        self.decoder = MODELS.build(decoder)

    def extract_feat(self, batch_inputs: torch.Tensor) -> torch.Tensor:
        """Directly extract features from the backbone."""
        if self.with_preprocessor:
            batch_inputs = self.preprocessor(batch_inputs)
        if self.with_backbone:
            batch_inputs = self.backbone(batch_inputs)
        return batch_inputs

    def loss(self, batch_inputs: torch.Tensor,
             batch_data_samples: Sequence[TextRecogDataSample],
             **kwargs) -> Dict:
        """Calculate losses from a batch of inputs and data samples.
        Args:
            inputs (tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            data_samples (list[TextRecogDataSample]): A list of N
                datasamples, containing meta information and gold
                annotations for each of the images.

        Returns:
            dict[str, tensor]: A dictionary of loss components.
        """
        feat = self.extract_feat(batch_inputs)
        out_enc = None
        if self.with_encoder:
            out_enc = self.encoder(feat, batch_data_samples)
        return self.decoder.loss(feat, out_enc, batch_data_samples)

    def predict(self, batch_inputs: torch.Tensor,
                batch_data_samples: SampleList, **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (torch.Tensor): Image input tensor.
            data_samples (list[TextRecogDataSample]): A list of N datasamples,
                containing meta information and gold annotations for each of
                the images.

        Returns:
            list[TextRecogDataSample]:  A list of N datasamples of prediction
            results. Results are stored in ``pred_text``.
        """
        feat = self.extract_feat(batch_inputs)
        out_enc = None
        if self.with_encoder:
            out_enc = self.encoder(feat, batch_data_samples)
        return self.decoder.predict(feat, out_enc, batch_data_samples)

    def _forward(self,
                 batch_inputs: torch.Tensor,
                 batch_data_samples: OptSampleList = None,
                 **kwargs):
        """Network forward process. Usually includes backbone, encoder and
        decoder forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[TextRecogDataSample]): A list of N
                datasamples, containing meta information and gold
                annotations for each of the images.

        Returns:
            Tensor: A tuple of features from ``decoder`` forward.
        """
        feat = self.extract_feat(batch_inputs)
        out_enc = None
        if self.with_encoder:
            out_enc = self.encoder(feat, batch_data_samples)
        return self.decoder(feat, out_enc, batch_data_samples)

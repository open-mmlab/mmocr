# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict

import torch

from mmocr.registry import MODELS
from mmocr.utils.typing import (ConfigType, InitConfigType, OptConfigType,
                                OptRecSampleList, RecForwardResults,
                                RecSampleList)
from .base import BaseRecognizer


@MODELS.register_module()
class EncoderDecoderRecognizer(BaseRecognizer):
    """Base class for encode-decode recognizer.

    Args:
        preprocessor (dict, optional): Config dict for preprocessor. Defaults
            to None.
        backbone (dict, optional): Backbone config. Defaults to None.
        encoder (dict, optional): Encoder config. If None, the output from
            backbone will be directly fed into ``decoder``. Defaults to None.
        decoder (dict, optional): Decoder config. Defaults to None.
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
                 preprocessor: OptConfigType = None,
                 backbone: OptConfigType = None,
                 encoder: OptConfigType = None,
                 decoder: OptConfigType = None,
                 data_preprocessor: ConfigType = None,
                 init_cfg: InitConfigType = None) -> None:

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

        # Decoder module
        assert decoder is not None
        self.decoder = MODELS.build(decoder)

    def extract_feat(self, inputs: torch.Tensor) -> torch.Tensor:
        """Directly extract features from the backbone."""
        if self.with_preprocessor:
            inputs = self.preprocessor(inputs)
        if self.with_backbone:
            inputs = self.backbone(inputs)
        return inputs

    def loss(self, inputs: torch.Tensor, data_samples: RecSampleList,
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
        feat = self.extract_feat(inputs)
        out_enc = None
        if self.with_encoder:
            out_enc = self.encoder(feat, data_samples)
        return self.decoder.loss(feat, out_enc, data_samples)

    def predict(self, inputs: torch.Tensor, data_samples: RecSampleList,
                **kwargs) -> RecSampleList:
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
        feat = self.extract_feat(inputs)
        out_enc = None
        if self.with_encoder:
            out_enc = self.encoder(feat, data_samples)
        return self.decoder.predict(feat, out_enc, data_samples)

    def _forward(self,
                 inputs: torch.Tensor,
                 data_samples: OptRecSampleList = None,
                 **kwargs) -> RecForwardResults:
        """Network forward process. Usually includes backbone, encoder and
        decoder forward without any post-processing.

         Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (list[TextRecogDataSample]): A list of N
                datasamples, containing meta information and gold
                annotations for each of the images.

        Returns:
            Tensor: A tuple of features from ``decoder`` forward.
        """
        feat = self.extract_feat(inputs)
        out_enc = None
        if self.with_encoder:
            out_enc = self.encoder(feat, data_samples)
        return self.decoder(feat, out_enc, data_samples)

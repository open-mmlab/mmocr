# Copyright (c) OpenMMLab. All rights reserved.

import warnings
from typing import Dict, Optional, Sequence

import torch

from mmocr.core.data_structures import TextRecogDataSample
from mmocr.registry import MODELS, TASK_UTILS
from .base import BaseRecognizer


@MODELS.register_module()
class EncodeDecodeRecognizer(BaseRecognizer):
    """Base class for encode-decode recognizer.

    Args:
        backbone (dict, optional): Backbone config. Defaults to None.
        encoder (dict, optional): Encoder config. If None, the output from
            backbone will be directly fed into ``decoder``. Defaults to None.
        decoder (dict, optional): Decoder config. Defaults to None.
        dictionary (dict, optional): Dictionary config. Defaults to None.
        max_seq_len (int): Maximum sequence length. Defaults to 40.
        preprocess_cfg (dict, optional): Model preprocessing config
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
                 preprocess_cfg: Dict = None,
                 init_cfg: Optional[Dict] = None) -> None:

        super().__init__(init_cfg=init_cfg, preprocess_cfg=preprocess_cfg)

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

    def extract_feat(self, inputs: torch.Tensor) -> torch.Tensor:
        """Directly extract features from the backbone."""
        if self.with_preprocessor:
            inputs = self.preprocessor(inputs)
        if self.with_backbone:
            inputs = self.backbone(inputs)
        return inputs

    def forward_train(self, inputs: torch.Tensor,
                      data_samples: Sequence[TextRecogDataSample],
                      **kwargs) -> Dict:
        """
            Args:
                inputs (tensor): Input images of shape (N, C, H, W).
                    Typically these should be mean centered and std scaled.
                data_samples (list[TextRecogDataSample]): A list of N
                    datasamples, containing meta information and gold
                    annotations for each of the images.

            Returns:
                dict[str, tensor]: A dictionary of loss components.
        """
        # TODO move to preprocess to update valid ratio
        super().forward_train(inputs, data_samples, **kwargs)
        for data_sample in data_samples:
            valid_ratio = data_sample.valid_ratio * data_sample.img_shape[
                1] / data_sample.batch_input_shape[1]
            data_sample.set_metainfo(dict(valid_ratio=valid_ratio))

        feat = self.extract_feat(inputs)
        out_enc = None
        if self.with_encoder:
            out_enc = self.encoder(feat, data_samples)
        data_samples = self.decoder.loss.get_targets(data_samples)
        out_dec = self.decoder(feat, out_enc, data_samples, train_mode=True)

        losses = self.decoder.loss(out_dec, data_samples)

        return losses

    def simple_test(self, inputs: torch.Tensor,
                    data_samples: Sequence[TextRecogDataSample],
                    **kwargs) -> Sequence[TextRecogDataSample]:
        """Test function without test-time augmentation.

        Args:
            inputs (torch.Tensor): Image input tensor.
            data_samples (list[TextRecogDataSample]): A list of N datasamples,
                containing meta information and gold annotations for each of
                the images.

        Returns:
            list[TextRecogDataSample]:  A list of N datasamples of prediction
            results. Results are stored in ``pred_text``.
        """
        # TODO move to preprocess to update valid ratio
        for data_sample in data_samples:
            valid_ratio = data_sample.valid_ratio * data_sample.img_shape[
                1] / data_sample.batch_input_shape[1]
            data_sample.set_metainfo(dict(valid_ratio=valid_ratio))
        feat = self.extract_feat(inputs)

        out_enc = None
        if self.with_encoder:
            out_enc = self.encoder(feat, data_samples)
        out_dec = self.decoder(feat, out_enc, data_samples, train_mode=False)
        data_samples = self.decoder.postprocessor(out_dec, data_samples)
        return data_samples

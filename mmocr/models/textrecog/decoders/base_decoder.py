# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Union

import torch
from mmcv.runner import BaseModule

from mmocr.core.data_structures import TextRecogDataSample
from mmocr.models.textrecog.dictionary import Dictionary
from mmocr.registry import MODELS, TASK_UTILS


@MODELS.register_module()
class BaseDecoder(BaseModule):
    """Base decoder for text recognition, build the loss and postprocessor.

    Args:
        dictionary (dict or :obj:`Dictionary`): The config for `Dictionary` or
            the instance of `Dictionary`.
        loss (dict, optional): Config to build loss. Defaults to None.
        postprocessor (dict, optional): Config to build postprocessor.
            Defaults to None.
        max_seq_len (int): Maximum sequence length. The
            sequence is usually generated from decoder. Defaults to 40.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """

    def __init__(self,
                 dictionary: Union[Dict, Dictionary],
                 loss_module: Optional[Dict] = None,
                 postprocessor: Optional[Dict] = None,
                 max_seq_len: int = 40,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        if isinstance(dictionary, dict):
            self.dictionary = TASK_UTILS.build(dictionary)
        elif isinstance(dictionary, Dictionary):
            self.dictionary = dictionary
        else:
            raise TypeError(
                'The type of dictionary should be `Dictionary` or dict, '
                f'but got {type(dictionary)}')
        self.loss_module = None
        self.postprocessor = None
        self.max_seq_len = max_seq_len

        if loss_module is not None:
            assert isinstance(loss_module, dict)
            loss_module.update(dictionary=dictionary)
            loss_module.update(max_seq_len=max_seq_len)
            self.loss_module = MODELS.build(loss_module)

        if postprocessor is not None:
            assert isinstance(postprocessor, dict)
            postprocessor.update(dictionary=dictionary)
            postprocessor.update(max_seq_len=max_seq_len)
            self.postprocessor = MODELS.build(postprocessor)

    def forward_train(
        self,
        feat: Optional[torch.Tensor] = None,
        out_enc: Optional[torch.Tensor] = None,
        data_samples: Optional[Sequence[TextRecogDataSample]] = None
    ) -> torch.Tensor:
        """Forward for training.

        Args:
            feat (torch.Tensor, optional): The feature map from backbone of
                shape :math:`(N, E, H, W)`. Defaults to None.
            out_enc (torch.Tensor, optional): Encoder output. Defaults to None.
            data_samples (Sequence[TextRecogDataSample]): Batch of
                TextRecogDataSample, containing gt_text information. Defaults
                to None.
        """
        raise NotImplementedError

    def forward_test(
        self,
        feat: Optional[torch.Tensor] = None,
        out_enc: Optional[torch.Tensor] = None,
        data_samples: Optional[Sequence[TextRecogDataSample]] = None
    ) -> torch.Tensor:
        """Forward for testing.

        Args:
            feat (torch.Tensor, optional): The feature map from backbone of
                shape :math:`(N, E, H, W)`. Defaults to None.
            out_enc (torch.Tensor, optional): Encoder output. Defaults to None.
            data_samples (Sequence[TextRecogDataSample]): Batch of
                TextRecogDataSample, containing gt_text information. Defaults
                to None.
        """
        raise NotImplementedError

    def loss(self,
             feat: Optional[torch.Tensor] = None,
             out_enc: Optional[torch.Tensor] = None,
             data_samples: Optional[Sequence[TextRecogDataSample]] = None
             ) -> Dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            feat (Tensor, optional): Features from the backbone. Defaults
                to None.
            out_enc (Tensor, optional): Features from the encoder.
                Defaults to None.
            data_samples (list[TextRecogDataSample], optional): A list of
                N datasamples, containing meta information and gold
                annotations for each of the images. Defaults to None.

        Returns:
            dict[str, tensor]: A dictionary of loss components.
        """
        out_dec = self(feat, out_enc, data_samples)
        return self.loss_module(out_dec, data_samples)

    def predict(
        self,
        feat: Optional[torch.Tensor] = None,
        out_enc: Optional[torch.Tensor] = None,
        data_samples: Optional[Sequence[TextRecogDataSample]] = None
    ) -> Sequence[TextRecogDataSample]:
        """Perform forward propagation of the decoder and postprocessor.

        Args:
            feat (Tensor, optional): Features from the backbone. Defaults
                to None.
            out_enc (Tensor, optional): Features from the encoder. Defaults
                to None.
            data_samples (list[TextRecogDataSample]): A list of N datasamples,
                containing meta information and gold annotations for each of
                the images. Defaults to None.

        Returns:
            list[TextRecogDataSample]:  A list of N datasamples of prediction
            results. Results are stored in ``pred_text``.
        """
        out_dec = self(feat, out_enc, data_samples)
        return self.postprocessor(out_dec, data_samples)

    def forward(
        self,
        feat: Optional[torch.Tensor] = None,
        out_enc: Optional[torch.Tensor] = None,
        data_samples: Optional[Sequence[TextRecogDataSample]] = None
    ) -> torch.Tensor:
        """Decoder forward.

         Args:
            feat (Tensor, optional): Features from the backbone. Defaults
                to None.
            out_enc (Tensor, optional): Features from the encoder.
                Defaults to None.
            data_samples (list[TextRecogDataSample]): A list of N datasamples,
                containing meta information and gold annotations for each of
                the images. Defaults to None.

        Returns:
            Tensor: Features from ``decoder`` forward.
        """
        if self.training:
            data_samples = self.loss_module.get_targets(data_samples)
            return self.forward_train(feat, out_enc, data_samples)
        else:
            return self.forward_test(feat, out_enc, data_samples)

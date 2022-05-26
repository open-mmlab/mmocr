# Copyright (c) OpenMMLab. All rights reserved.
import copy
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
from mmcv.runner import BaseModule, auto_fp16
from mmdet.core.utils import stack_batch

from mmocr.core.data_structures import TextRecogDataSample


class BaseRecognizer(BaseModule, metaclass=ABCMeta):
    """Base class for text recognition.

    Args:
        preprocess_cfg (dict, optional): Model preprocessing config
            for processing the input image data. Keys allowed are
            ``to_rgb``(bool), ``pad_size_divisor``(int), ``pad_value``(int or
            float), ``mean``(int or float) and ``std``(int or float).
            Preprcessing order: 1. to rgb; 2. normalization 3. pad.
            Defaults to None.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 preprocess_cfg: Optional[Dict] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.fp16_enabled = False
        self.preprocess_cfg = preprocess_cfg

        self.pad_size_divisor = 0
        self.pad_value = 0

        if self.preprocess_cfg is not None:
            assert isinstance(self.preprocess_cfg, dict)
            self.preprocess_cfg = copy.deepcopy(self.preprocess_cfg)

            self.to_rgb = preprocess_cfg.get('to_rgb', False)
            self.pad_size_divisor = preprocess_cfg.get('pad_size_divisor', 0)
            self.pad_value = preprocess_cfg.get('pad_value', 0)
            self.register_buffer(
                'pixel_mean',
                torch.tensor(preprocess_cfg['mean']).view(-1, 1, 1), False)
            self.register_buffer(
                'pixel_std',
                torch.tensor(preprocess_cfg['std']).view(-1, 1, 1), False)
        else:
            # Only used to provide device information
            self.register_buffer('pixel_mean', torch.tensor(1), False)

    @property
    def device(self) -> torch.device:
        return self.pixel_mean.device

    @property
    def with_backbone(self):
        """bool: whether the recognizer has a backbone"""
        return getattr(self, 'backbone', None) is not None

    @property
    def with_encoder(self):
        """bool: whether the recognizer has an encoder"""
        return getattr(self, 'encoder', None) is not None

    @property
    def with_preprocessor(self):
        """bool: whether the recognizer has a preprocessor"""
        return getattr(self, 'preprocessor', None) is not None

    @property
    def with_dictionary(self):
        """bool: whether the recognizer has a dictionary"""
        return getattr(self, 'dictionary', None) is not None

    @property
    def with_decoder(self):
        """bool: whether the recognizer has a decoder"""
        return getattr(self, 'decoder', None) is not None

    @abstractmethod
    def extract_feat(self, inputs: torch.Tensor) -> torch.Tensor:
        """Extract features from images."""
        pass

    @auto_fp16(apply_to=('inputs', ))
    def forward_train(self, inputs: torch.Tensor,
                      data_samples: Sequence[TextRecogDataSample],
                      **kwargs) -> Dict:
        """Training function.

        Args:
            inputs (Tensor):The image Tensor should have a shape NxCxHxW.
                These should usually be mean centered and std scaled.
            data_samples (list[:obj:`TextRecogDataSample`]):The batch
                data samples. It usually includes ``gt_text`` information.
        """
        # TODO: maybe remove to stack_batch
        # NOTE the batched image size information may be useful for
        # calculating valid ratio.
        batch_input_shape = tuple(inputs[0].size()[-2:])
        for data_sample in data_samples:
            data_sample.set_metainfo({'batch_input_shape': batch_input_shape})

    @abstractmethod
    def simple_test(self, inputs: torch.Tensor,
                    data_samples: Sequence[TextRecogDataSample],
                    **kwargs) -> Sequence[TextRecogDataSample]:
        pass

    def aug_test(self, imgs: torch.Tensor,
                 data_samples: Sequence[Sequence[TextRecogDataSample]],
                 **kwargs):
        """Test function with test time augmentation."""
        pass

    @auto_fp16(apply_to=('img', ))
    def forward(self,
                data: Sequence[Dict],
                optimizer: Optional[Union[torch.optim.Optimizer, Dict]] = None,
                return_loss: bool = False,
                **kwargs):
        """The iteration step during training and testing. This method defines
        an iteration step during training and testing, except for the back
        propagation and optimizer updating during training, which are done in
        an optimizer hook.

        Args:
            data (list[dict]): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` or dict, optional): The
                optimizer of runner. This argument is unused and reserved.
                Defaults to None.
            return_loss (bool): Whether to return loss. In general,
                it will be set to True during training and False
                during testing. Defaults to False.

        Returns:
            During training
                dict: It should contain at least 3 keys: ``loss``,
                ``log_vars``, ``num_samples``.
                    - ``loss`` is a tensor for back propagation, which can be a
                      weighted sum of multiple losses.
                    - ``log_vars`` contains all the variables to be sent to the
                        logger.
                    - ``num_samples`` indicates the batch size (when the model
                        is DDP, it means the batch size on each GPU), which is
                        used for averaging the logs.

            During testing
                list(obj:`TextRecogDataSample`): Recognition results of the
                input images. Each TextRecogDataSample usually contains
                ``pred_text``
        """

        inputs, data_samples = self.preprocss_data(data)

        if return_loss:
            losses = self.forward_train(inputs, data_samples, **kwargs)
            loss, log_vars = self._parse_losses(losses)

            outputs = dict(
                loss=loss, log_vars=log_vars, num_samples=len(data_samples))
            return outputs
        else:
            # TODO: refactor and support aug test later
            assert isinstance(data[0]['inputs'], torch.Tensor), \
                'Only support simple test currently. Aug-test is ' \
                'not supported yet'
            return self.forward_simple_test(inputs, data_samples, **kwargs)

    def _parse_losses(self, losses: Dict) -> Tuple[torch.Tensor, Dict]:
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw outputs of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def preprocss_data(self, data: List[Dict]) -> Tuple:
        """ Process input data during training and simple testing phases.
        Args:
            data (list[dict]): The data to be processed, which
                comes from dataloader.

        Returns:
            tuple:  It should contain 2 items.

              - inputs (Tensor): The batch input tensor.
              - data_samples (list[:obj:`TextRecogDataSample`]): The
                Data Samples. It usually includes `gt_text` information.
        """
        inputs = [data_['inputs'] for data_ in data]
        data_samples = [data_['data_sample'] for data_ in data]

        data_samples = [
            data_sample.to(self.device) for data_sample in data_samples
        ]
        inputs = [_input.to(self.device) for _input in inputs]

        if self.preprocess_cfg is None:
            return stack_batch(inputs).float(), data_samples

        if self.to_rgb and inputs[0].size(0) == 3:
            inputs = [_input[[2, 1, 0], ...] for _input in inputs]
        inputs = [(_input - self.pixel_mean) / self.pixel_std
                  for _input in inputs]
        inputs = stack_batch(inputs, self.pad_size_divisor, self.pad_value)
        return inputs, data_samples

    @auto_fp16(apply_to=('inputs', ))
    def forward_simple_test(self, inputs: torch.Tensor,
                            data_samples: Sequence[TextRecogDataSample],
                            **kwargs) -> Sequence[TextRecogDataSample]:
        """
        Args:
            inputs (Tensor): The input Tensor should have a
                shape NxCxHxW.
            data_samples (list[:obj:`TextRecogDataSample`]): The Data
                Samples. It usually includes ``gt_text`` information.

        Returns:
            list[obj:`TextRecogDataSample`]: Detection results of the
            input images. Each TextRecogDataSample usually contains
            ``pred_text``.
        """
        # TODO: Consider merging with forward_train logic
        batch_input_shape = tuple(inputs[0].size()[-2:])
        for data_sample in data_samples:
            data_sample.set_metainfo({'batch_input_shape': batch_input_shape})

        return self.simple_test(inputs, data_samples, **kwargs)

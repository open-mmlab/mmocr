import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import mmcv
import torch
import torch.distributed as dist
import torch.nn as nn
from mmcv.runner import auto_fp16
from mmcv.utils import print_log

from mmocr.core import imshow_text_label
from mmocr.utils import get_root_logger


class BaseRecognizer(nn.Module, metaclass=ABCMeta):
    """Base class for text recognition."""

    def __init__(self):
        super().__init__()
        self.fp16_enabled = False

    @abstractmethod
    def extract_feat(self, imgs):
        """Extract features from images."""
        pass

    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):
        """
        Args:
            img (tensor): tensors with shape (N, C, H, W).
                Typically should be mean centered and std scaled.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details of the values of these keys, see
                :class:`mmdet.datasets.pipelines.Collect`.
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        pass

    @abstractmethod
    def simple_test(self, img, img_metas, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation.

        Args:
            imgs (list[tensor]): Tensor should have shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): The metadata of images.
        """
        pass

    def init_weights(self, pretrained=None):
        """Initialize the weights for detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if pretrained is not None:
            logger = get_root_logger()
            print_log(f'load model from: {pretrained}', logger=logger)

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (tensor | list[tensor]): Tensor should have shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[dict] | list[list[dict]]):
                The outer list indicates images in a batch.
        """
        if isinstance(imgs, list):
            assert len(imgs) == len(img_metas)
            assert len(imgs) > 0
            assert imgs[0].size(0) == 1, ('aug test does not support '
                                          f'inference with batch size '
                                          f'{imgs[0].size(0)}')
            return self.aug_test(imgs, img_metas, **kwargs)

        return self.simple_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note that img and img_meta are single-nested (i.e. tensor and
        list[dict]).
        """

        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)

        if isinstance(img, list):
            for idx, i in enumerate(img):
                if i.dim() == 3:
                    img[idx] = i.unsqueeze(0)
        else:
            img_metas = img_metas[0]

        return self.forward_test(img, img_metas, **kwargs)

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw outputs of the network, which usually contain
                losses and other necessary infomation.

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

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer update, which are done by an optimizer
        hook. Note that in some complicated cases or models (e.g. GAN),
        the whole process (including the back propagation and optimizer update)
        is also defined by this method.

        Args:
            data (dict): The outputs of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which is a
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size used for
                averaging the logs (Note: for the
                DDP model, num_samples refers to the batch size for each GPU).
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def val_step(self, data, optimizer):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but is
        used during val epochs. Note that the evaluation after training epochs
        is not implemented by this method, but by an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def show_result(self,
                    img,
                    result,
                    gt_label='',
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None,
                    **kwargs):
        """Draw `result` on `img`.

        Args:
            img (str or tensor): The image to be displayed.
            result (dict): The results to draw on `img`.
            gt_label (str): Ground truth label of img.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The output filename.
                Default: None.

        Returns:
            img (tensor): Only if not `show` or `out_file`.
        """
        img = mmcv.imread(img)
        img = img.copy()
        pred_label = None
        if 'text' in result.keys():
            pred_label = result['text']

        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw text label
        if pred_label is not None:
            img = imshow_text_label(
                img,
                pred_label,
                gt_label,
                show=show,
                win_name=win_name,
                wait_time=wait_time,
                out_file=out_file)

        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only '
                          'result image will be returned')
            return img

        return img

# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch.nn as nn
from mmcv.cnn import ACTIVATION_LAYERS as MMCV_ACTIVATION_LAYERS
from mmcv.cnn import UPSAMPLE_LAYERS as MMCV_UPSAMPLE_LAYERS
from mmcv.utils import Registry, build_from_cfg
from mmdet.models.builder import BACKBONES as MMDET_BACKBONES

RECOGNIZERS = Registry('recognizer')
CONVERTORS = Registry('convertor')
ENCODERS = Registry('encoder')
DECODERS = Registry('decoder')
PREPROCESSOR = Registry('preprocessor')
POSTPROCESSOR = Registry('postprocessor')

UPSAMPLE_LAYERS = Registry('upsample layer', parent=MMCV_UPSAMPLE_LAYERS)
BACKBONES = Registry('models', parent=MMDET_BACKBONES)
LOSSES = BACKBONES
DETECTORS = BACKBONES
ROI_EXTRACTORS = BACKBONES
HEADS = BACKBONES
NECKS = BACKBONES
FUSERS = BACKBONES

ACTIVATION_LAYERS = Registry('activation layer', parent=MMCV_ACTIVATION_LAYERS)


def build_recognizer(cfg, train_cfg=None, test_cfg=None):
    """Build recognizer."""
    return build_from_cfg(cfg, RECOGNIZERS,
                          dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_convertor(cfg):
    """Build label convertor for scene text recognizer."""
    return build_from_cfg(cfg, CONVERTORS)


def build_encoder(cfg):
    """Build encoder for scene text recognizer."""
    return build_from_cfg(cfg, ENCODERS)


def build_decoder(cfg):
    """Build decoder for scene text recognizer."""
    return build_from_cfg(cfg, DECODERS)


def build_preprocessor(cfg):
    """Build preprocessor for scene text recognizer."""
    return build_from_cfg(cfg, PREPROCESSOR)


def build_postprocessor(cfg, train_cfg=None, test_cfg=None):
    """Build postprocessor for ocr."""
    return build_from_cfg(cfg, POSTPROCESSOR,
                          dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_roi_extractor(cfg):
    """Build roi extractor."""
    return ROI_EXTRACTORS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_fuser(cfg):
    """Build fuser."""
    return FUSERS.build(cfg)


def build_upsample_layer(cfg, *args, **kwargs):
    """Build upsample layer.

    Args:
        cfg (dict): The upsample layer config, which should contain:

            - type (str): Layer type.
            - scale_factor (int): Upsample ratio, which is not applicable to
                deconv.
            - layer args: Args needed to instantiate a upsample layer.
        args (argument list): Arguments passed to the ``__init__``
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the
            ``__init__`` method of the corresponding conv layer.

    Returns:
        nn.Module: Created upsample layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        raise KeyError(
            f'the cfg dict must contain the key "type", but got {cfg}')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in UPSAMPLE_LAYERS:
        raise KeyError(f'Unrecognized upsample type {layer_type}')
    else:
        upsample = UPSAMPLE_LAYERS.get(layer_type)

    if upsample is nn.Upsample:
        cfg_['mode'] = layer_type
    layer = upsample(*args, **kwargs, **cfg_)
    return layer


def build_activation_layer(cfg):
    """Build activation layer.

    Args:
        cfg (dict): The activation layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an activation layer.

    Returns:
        nn.Module: Created activation layer.
    """
    return build_from_cfg(cfg, ACTIVATION_LAYERS)


def build_detector(cfg, train_cfg=None, test_cfg=None):
    """Build detector."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return DETECTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))

# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch.nn as nn

from mmocr.registry import MODELS

CONVERTORS = MODELS
ENCODERS = MODELS
DECODERS = MODELS
PREPROCESSOR = MODELS
POSTPROCESSOR = MODELS

UPSAMPLE_LAYERS = MODELS
BACKBONES = MODELS
LOSSES = MODELS
DETECTORS = MODELS
ROI_EXTRACTORS = MODELS
HEADS = MODELS
NECKS = MODELS
FUSERS = MODELS
RECOGNIZERS = MODELS

ACTIVATION_LAYERS = MODELS


def build_recognizer(cfg, train_cfg=None, test_cfg=None):
    """Build recognizer."""
    warnings.warn('``build_recognizer`` would be deprecated soon, please use '
                  '``mmocr.registry.MODELS.build()`` ')
    return RECOGNIZERS(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_convertor(cfg):
    """Build label convertor for scene text recognizer."""
    warnings.warn('``build_convertor`` would be deprecated soon, please use '
                  '``mmocr.registry.MODELS.build()`` ')
    return CONVERTORS.build(cfg)


def build_encoder(cfg):
    """Build encoder for scene text recognizer."""
    warnings.warn('``build_encoder`` would be deprecated soon, please use '
                  '``mmocr.registry.MODELS.build()`` ')
    return ENCODERS.build(cfg)


def build_decoder(cfg):
    """Build decoder for scene text recognizer."""
    warnings.warn('``build_decoder`` would be deprecated soon, please use '
                  '``mmocr.registry.MODELS.build()`` ')
    return DECODERS.build(cfg)


def build_preprocessor(cfg):
    """Build preprocessor for scene text recognizer."""
    warnings.warn(
        '``build_preprocessor`` would be deprecated soon, please use '
        '``mmocr.registry.MODELS.build()`` ')
    return PREPROCESSOR(cfg)


def build_postprocessor(cfg):
    """Build postprocessor for scene text detector."""
    warnings.warn(
        '``build_postprocessor`` would be deprecated soon, please use '
        '``mmocr.registry.MODELS.build()`` ')
    return POSTPROCESSOR.build(cfg)


def build_roi_extractor(cfg):
    """Build roi extractor."""
    warnings.warn(
        '``build_roi_extractor`` would be deprecated soon, please use '
        '``mmocr.registry.MODELS.build()`` ')
    return ROI_EXTRACTORS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    warnings.warn('``build_loss`` would be deprecated soon, please use '
                  '``mmocr.registry.MODELS.build()`` ')
    return LOSSES.build(cfg)


def build_backbone(cfg):
    """Build backbone."""
    warnings.warn('``build_backbone`` would be deprecated soon, please use '
                  '``mmocr.registry.MODELS.build()`` ')
    return BACKBONES.build(cfg)


def build_head(cfg):
    """Build head."""
    warnings.warn('``build_head`` would be deprecated soon, please use '
                  '``mmocr.registry.MODELS.build()`` ')
    return HEADS.build(cfg)


def build_neck(cfg):
    """Build neck."""
    warnings.warn('``build_neck`` would be deprecated soon, please use '
                  '``mmocr.registry.MODELS.build()`` ')
    return NECKS.build(cfg)


def build_fuser(cfg):
    """Build fuser."""
    warnings.warn('``build_fuser`` would be deprecated soon, please use '
                  '``mmocr.registry.MODELS.build()`` ')
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
    warnings.warn(
        '``build_activation_layer`` would be deprecated soon, please use '
        '``mmocr.registry.MODELS.build()`` ')
    return ACTIVATION_LAYERS.build(cfg)


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

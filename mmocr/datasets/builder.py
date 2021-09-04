# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import Registry, build_from_cfg

LOADERS = Registry('loader')
PARSERS = Registry('parser')


def build_loader(cfg):
    """Build anno file loader."""
    return build_from_cfg(cfg, LOADERS)


def build_parser(cfg):
    """Build anno file parser."""
    return build_from_cfg(cfg, PARSERS)

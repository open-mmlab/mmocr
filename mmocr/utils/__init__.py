# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import Registry, build_from_cfg

from .box_util import (bezier_to_polygon, is_on_same_line, sort_points,
                       stitch_boxes_into_lines)
from .check_argument import (equal_len, is_2dlist, is_3dlist, is_none_or_type,
                             is_type_list, valid_boundary)
from .collect_env import collect_env
from .config import (disable_text_recog_aug_test, replace_image_to_tensor,
                     unify_recog_pipeline)
from .data_convert_util import convert_annotations
from .fileio import list_from_file, list_to_file
from .img_util import drop_orientation, is_not_png
from .lmdb_util import lmdb_converter
from .logger import get_root_logger
from .model import revert_sync_batchnorm
from .setup_env import setup_multi_processes
from .string_util import StringStrip

__all__ = [
    'Registry', 'build_from_cfg', 'get_root_logger', 'collect_env',
    'is_3dlist', 'is_type_list', 'is_none_or_type', 'equal_len', 'is_2dlist',
    'valid_boundary', 'lmdb_converter', 'drop_orientation',
    'convert_annotations', 'is_not_png', 'list_to_file', 'list_from_file',
    'is_on_same_line', 'stitch_boxes_into_lines', 'StringStrip',
    'revert_sync_batchnorm', 'bezier_to_polygon', 'sort_points',
    'setup_multi_processes', 'unify_recog_pipeline', 'replace_image_to_tensor',
    'disable_text_recog_aug_test'
]

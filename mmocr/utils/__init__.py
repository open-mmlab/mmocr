# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import Registry, build_from_cfg

from .api_utils import disable_text_recog_aug_test
from .bbox_utils import (bbox2poly, box_center_distance, box_diag,
                         rescale_bboxes, sort_vertex, sort_vertex8)
from .box_util import (bezier_to_polygon, is_on_same_line, sort_points,
                       stitch_boxes_into_lines)
from .check_argument import (equal_len, is_2dlist, is_3dlist, is_none_or_type,
                             is_type_list, valid_boundary)
from .collect_env import collect_env
from .data_convert_util import dump_ocr_data, recog_anno_to_imginfo
from .evaluation_utils import (compute_hmean, filter_2dlist_result,
                               many2one_match_ic13, one2one_match_ic13,
                               select_top_boundary)
from .fileio import list_from_file, list_to_file
from .lmdb_util import recog2lmdb
from .logger import get_root_logger
from .model import revert_sync_batchnorm
from .point_utils import dist_points2line, point_distance, points_center
from .polygon_utils import (boundary_iou, crop_polygon, is_poly_inside_rect,
                            offset_polygon, poly2bbox, poly2shapely,
                            poly_intersection, poly_iou, poly_make_valid,
                            poly_union, polys2shapely, rescale_polygon,
                            rescale_polygons)
from .setup_env import register_all_modules
from .string_util import StringStrip

__all__ = [
    'Registry', 'build_from_cfg', 'get_root_logger', 'collect_env',
    'is_3dlist', 'is_type_list', 'is_none_or_type', 'equal_len', 'is_2dlist',
    'valid_boundary', 'list_to_file', 'list_from_file', 'is_on_same_line',
    'stitch_boxes_into_lines', 'StringStrip', 'revert_sync_batchnorm',
    'bezier_to_polygon', 'sort_points', 'recog2lmdb', 'dump_ocr_data',
    'recog_anno_to_imginfo', 'rescale_polygons', 'rescale_polygon',
    'rescale_bboxes', 'bbox2poly', 'crop_polygon', 'is_poly_inside_rect',
    'poly2bbox', 'poly_intersection', 'poly_iou', 'poly_make_valid',
    'poly_union', 'poly2shapely', 'polys2shapely', 'register_all_modules',
    'dist_points2line', 'offset_polygon', 'sort_vertex8', 'sort_vertex',
    'disable_text_recog_aug_test', 'box_center_distance', 'box_diag',
    'compute_hmean', 'filter_2dlist_result', 'many2one_match_ic13',
    'one2one_match_ic13', 'select_top_boundary', 'boundary_iou',
    'point_distance', 'points_center'
]

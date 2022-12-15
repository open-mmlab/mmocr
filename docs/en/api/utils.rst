.. role:: hidden
    :class: hidden-section

mmocr.utils
===================================

.. contents:: mmocr.utils
   :depth: 2
   :local:
   :backlinks: top

.. currentmodule:: mmocr.utils

Image Utils
---------------------------------------------
.. autosummary::
   :toctree: generated
   :nosignatures:
   crop_img
   warp_img


Box Utils
---------------------------------------------
.. autosummary::
   :toctree: generated
   :nosignatures:

   bbox2poly
   bbox_center_distance
   bbox_diag_distance
   bezier2polygon
   is_on_same_line
   rescale_bboxes

   stitch_boxes_into_lines


Point Utils
---------------------------------------------
.. autosummary::
   :toctree: generated
   :nosignatures:

   point_distance
   points_center

Polygon Utils
---------------------------------------------
.. autosummary::
   :toctree: generated
   :nosignatures:

   boundary_iou
   crop_polygon
   is_poly_inside_rect
   offset_polygon
   poly2bbox
   poly2shapely
   poly_intersection
   poly_iou
   poly_make_valid
   poly_union
   polys2shapely
   rescale_polygon
   rescale_polygons
   shapely2poly
   sort_points
   sort_vertex
   sort_vertex8


Mask Utils
---------------------------------------------
.. autosummary::
   :toctree: generated
   :nosignatures:

   fill_hole


Misc Utils
---------------------------------------------
.. autosummary::
   :toctree: generated
   :nosignatures:

   equal_len
   is_2dlist
   is_3dlist
   is_none_or_type
   is_type_list


Setup Env
---------------------------------------------
.. autosummary::
   :toctree: generated
   :nosignatures:

   register_all_modules

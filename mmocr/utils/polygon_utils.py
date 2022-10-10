# Copyright (c) OpenMMLab. All rights reserved.
import functools
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pyclipper
import shapely
from mmengine.utils import is_list_of
from shapely.geometry import MultiPolygon, Polygon

from mmocr.utils import bbox2poly, valid_boundary
from mmocr.utils.check_argument import is_2dlist
from mmocr.utils.typing import ArrayLike


def rescale_polygon(polygon: ArrayLike,
                    scale_factor: Tuple[int, int],
                    mode: str = 'mul') -> np.ndarray:
    """Rescale a polygon according to scale_factor.

    The behavior is different depending on the mode. When mode is 'mul', the
    coordinates will be multiplied by scale_factor, which is usually used in
    preprocessing transforms such as :func:`Resize`.
    The coordinates will be divided by scale_factor if mode is 'div'. It can be
    used in postprocessors to recover the polygon in the original
    image size.

    Args:
        polygon (ArrayLike): A polygon. In any form can be converted
            to an 1-D numpy array. E.g. list[float], np.ndarray,
            or torch.Tensor. Polygon is written in
            [x1, y1, x2, y2, ...].
        scale_factor (tuple(int, int)): (w_scale, h_scale).
        model (str): Rescale mode. Can be 'mul' or 'div'. Defaults to 'mul'.

    Returns:
        np.ndarray: Rescaled polygon.
    """
    assert len(polygon) % 2 == 0
    assert mode in ['mul', 'div']
    polygon = np.array(polygon, dtype=np.float32)
    poly_shape = polygon.shape
    reshape_polygon = polygon.reshape(-1, 2)
    scale_factor = np.array(scale_factor, dtype=float)
    if mode == 'div':
        scale_factor = 1 / scale_factor
    polygon = (reshape_polygon * scale_factor[None]).reshape(poly_shape)
    return polygon


def rescale_polygons(polygons: Sequence[ArrayLike],
                     scale_factor: Tuple[int, int],
                     mode: str = 'mul') -> Sequence[np.ndarray]:
    """Rescale polygons according to scale_factor.

    The behavior is different depending on the mode. When mode is 'mul', the
    coordinates will be multiplied by scale_factor, which is usually used in
    preprocessing transforms such as :func:`Resize`.
    The coordinates will be divided by scale_factor if mode is 'div'. It can be
    used in postprocessors to recover the polygon in the original
    image size.

    Args:
        polygons (list[ArrayLike]): A list of polygons, each written in
            [x1, y1, x2, y2, ...] and in any form can be converted
            to an 1-D numpy array. E.g. list[list[float]],
            list[np.ndarray], or list[torch.Tensor].
        scale_factor (tuple(int, int)): (w_scale, h_scale).
        model (str): Rescale mode. Can be 'mul' or 'div'. Defaults to 'mul'.

    Returns:
        list[np.ndarray]: Rescaled polygons.
    """
    results = []
    for polygon in polygons:
        results.append(rescale_polygon(polygon, scale_factor, mode))
    return results


def poly2bbox(polygon: ArrayLike) -> np.array:
    """Converting a polygon to a bounding box.

    Args:
         polygon (ArrayLike): A polygon. In any form can be converted
             to an 1-D numpy array. E.g. list[float], np.ndarray,
             or torch.Tensor. Polygon is written in
             [x1, y1, x2, y2, ...].

     Returns:
         np.array: The converted bounding box [x1, y1, x2, y2]
    """
    assert len(polygon) % 2 == 0
    polygon = np.array(polygon, dtype=np.float32)
    x = polygon[::2]
    y = polygon[1::2]
    return np.array([min(x), min(y), max(x), max(y)])


def poly2shapely(polygon: ArrayLike) -> Polygon:
    """Convert a polygon to shapely.geometry.Polygon.

    Args:
        polygon (ArrayLike): A set of points of 2k shape.

    Returns:
        polygon (Polygon): A polygon object.
    """
    polygon = np.array(polygon, dtype=np.float32)
    assert polygon.size % 2 == 0 and polygon.size >= 6

    polygon = polygon.reshape([-1, 2])
    return Polygon(polygon)


def polys2shapely(polygons: Sequence[ArrayLike]) -> Sequence[Polygon]:
    """Convert a nested list of boundaries to a list of Polygons.

    Args:
        polygons (list): The point coordinates of the instance boundary.

    Returns:
        list: Converted shapely.Polygon.
    """
    return [poly2shapely(polygon) for polygon in polygons]


def shapely2poly(polygon: Polygon) -> np.array:
    """Convert a nested list of boundaries to a list of Polygons.

    Args:
        polygon (Polygon): A polygon represented by shapely.Polygon.

    Returns:
        np.array: Converted numpy array
    """
    return np.array(polygon.exterior.coords).reshape(-1, )


def crop_polygon(polygon: ArrayLike,
                 crop_box: np.ndarray) -> Union[np.ndarray, None]:
    """Crop polygon to be within a box region.

    Args:
        polygon (ndarray): polygon in shape (N, ).
        crop_box (ndarray): target box region in shape (4, ).

    Returns:
        np.array or None: Cropped polygon. If the polygon is not within the
            crop box, return None.
    """
    poly = poly2shapely(polygon)
    crop_poly = poly2shapely(bbox2poly(crop_box))
    poly_cropped = poly.intersection(crop_poly)
    if poly_cropped.area == 0. or not isinstance(
            poly_cropped, shapely.geometry.polygon.Polygon):
        # If polygon is outside crop_box region or the intersection is not a
        # polygon, return None.
        return None
    else:
        poly_cropped = np.array(poly_cropped.boundary.xy, dtype=np.float32)
        poly_cropped = poly_cropped[:, :-1].T
        # reverse poly_cropped to have clockwise order
        poly_cropped = poly_cropped[::-1, :].reshape(-1)
        return poly_cropped


def poly_make_valid(poly: Polygon) -> Polygon:
    """Convert a potentially invalid polygon to a valid one by eliminating
    self-crossing or self-touching parts.

    Args:
        poly (Polygon): A polygon needed to be converted.

    Returns:
        Polygon: A valid polygon.
    """
    assert isinstance(poly, Polygon)
    return poly if poly.is_valid else poly.buffer(0)


def poly_intersection(poly_a: Polygon,
                      poly_b: Polygon,
                      invalid_ret: Optional[Union[float, int]] = None,
                      return_poly: bool = False
                      ) -> Tuple[float, Optional[Polygon]]:
    """Calculate the intersection area between two polygons.

    Args:
        poly_a (Polygon): Polygon a.
        poly_b (Polygon): Polygon b.
        invalid_ret (float or int, optional): The return value when the
            invalid polygon exists. If it is not specified, the function
            allows the computation to proceed with invalid polygons by
            cleaning the their self-touching or self-crossing parts.
            Defaults to None.
        return_poly (bool): Whether to return the polygon of the intersection
            Defaults to False.

    Returns:
        float or tuple(float, Polygon): Returns the intersection area or
        a tuple ``(area, Optional[poly_obj])``, where the `area` is the
        intersection area between two polygons and `poly_obj` is The Polygon
        object of the intersection area. Set as `None` if the input is invalid.
        Set as `None` if the input is invalid. `poly_obj` will be returned
        only if `return_poly` is `True`.
    """
    assert isinstance(poly_a, Polygon)
    assert isinstance(poly_b, Polygon)
    assert invalid_ret is None or isinstance(invalid_ret, (float, int))

    if invalid_ret is None:
        poly_a = poly_make_valid(poly_a)
        poly_b = poly_make_valid(poly_b)

    poly_obj = None
    area = invalid_ret
    if poly_a.is_valid and poly_b.is_valid:
        poly_obj = poly_a.intersection(poly_b)
        area = poly_obj.area
    return (area, poly_obj) if return_poly else area


def poly_union(
    poly_a: Polygon,
    poly_b: Polygon,
    invalid_ret: Optional[Union[float, int]] = None,
    return_poly: bool = False
) -> Tuple[float, Optional[Union[Polygon, MultiPolygon]]]:
    """Calculate the union area between two polygons.

    Args:
        poly_a (Polygon): Polygon a.
        poly_b (Polygon): Polygon b.
        invalid_ret (float or int, optional): The return value when the
            invalid polygon exists. If it is not specified, the function
            allows the computation to proceed with invalid polygons by
            cleaning the their self-touching or self-crossing parts.
            Defaults to False.
        return_poly (bool): Whether to return the polygon of the union.
            Defaults to False.

    Returns:
        tuple: Returns a tuple ``(area, Optional[poly_obj])``, where
        the `area` is the union between two polygons and `poly_obj` is the
        Polygon or MultiPolygon object of the union of the inputs. The type
        of object depends on whether they intersect or not. Set as `None`
        if the input is invalid. `poly_obj` will be returned only if
        `return_poly` is `True`.
    """
    assert isinstance(poly_a, Polygon)
    assert isinstance(poly_b, Polygon)
    assert invalid_ret is None or isinstance(invalid_ret, (float, int))

    if invalid_ret is None:
        poly_a = poly_make_valid(poly_a)
        poly_b = poly_make_valid(poly_b)

    poly_obj = None
    area = invalid_ret
    if poly_a.is_valid and poly_b.is_valid:
        poly_obj = poly_a.union(poly_b)
        area = poly_obj.area
    return (area, poly_obj) if return_poly else area


def poly_iou(poly_a: Polygon,
             poly_b: Polygon,
             zero_division: float = 0.) -> float:
    """Calculate the IOU between two polygons.

    Args:
        poly_a (Polygon): Polygon a.
        poly_b (Polygon): Polygon b.
        zero_division (float): The return value when invalid polygon exists.

    Returns:
        float: The IoU between two polygons.
    """
    assert isinstance(poly_a, Polygon)
    assert isinstance(poly_b, Polygon)
    area_inters = poly_intersection(poly_a, poly_b)
    area_union = poly_union(poly_a, poly_b)
    return area_inters / area_union if area_union != 0 else zero_division


def is_poly_inside_rect(poly: ArrayLike, rect: np.ndarray) -> bool:
    """Check if the polygon is inside the target region.
        Args:
            poly (ArrayLike): Polygon in shape (N, ).
            rect (ndarray): Target region [x1, y1, x2, y2].

        Returns:
            bool: Whether the polygon is inside the cropping region.
        """

    poly = poly2shapely(poly)
    rect = poly2shapely(bbox2poly(rect))
    return rect.contains(poly)


def offset_polygon(poly: ArrayLike, distance: float) -> ArrayLike:
    """Offset (expand/shrink) the polygon by the target distance. It's a
    wrapper around pyclipper based on Vatti clipping algorithm.

    Warning:
        Polygon coordinates will be casted to int type in PyClipper. Mind the
        potential precision loss caused by the casting.

    Args:
        poly (ArrayLike): A polygon. In any form can be converted
            to an 1-D numpy array. E.g. list[float], np.ndarray,
            or torch.Tensor. Polygon is written in
            [x1, y1, x2, y2, ...].
        distance (float): The offset distance. Positive value means expanding,
            negative value means shrinking.

    Returns:
        np.array: 1-D Offsetted polygon ndarray in float32 type. If the
        result polygon is invalid or has been split into several parts,
        return an empty array.
    """
    poly = np.array(poly).reshape(-1, 2)
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    # Returned result will be in type of int32, convert it back to float32
    # following MMOCR's convention
    result = np.array(pco.Execute(distance))
    if len(result) > 0 and isinstance(result[0], list):
        # The processed polygon has been split into several parts
        result = np.array([])
    result = result.astype(np.float32)
    # Always use the first polygon since only one polygon is expected
    # But when the resulting polygon is invalid, return the empty array
    # as it is
    return result if len(result) == 0 else result[0].flatten()


def boundary_iou(src: List,
                 target: List,
                 zero_division: Union[int, float] = 0) -> float:
    """Calculate the IOU between two boundaries.

    Args:
       src (list): Source boundary.
       target (list): Target boundary.
       zero_division (int or float): The return value when invalid
                                    boundary exists.

    Returns:
       float: The iou between two boundaries.
    """
    assert valid_boundary(src, False)
    assert valid_boundary(target, False)
    src_poly = poly2shapely(src)
    target_poly = poly2shapely(target)

    return poly_iou(src_poly, target_poly, zero_division=zero_division)


def sort_points(points):
    # TODO Add typehints & test & docstring
    """Sort arbitory points in clockwise order. Reference:
    https://stackoverflow.com/a/6989383.

    Args:
        points (list[ndarray] or ndarray or list[list]): A list of unsorted
            boundary points.

    Returns:
        list[ndarray]: A list of points sorted in clockwise order.
    """

    assert is_list_of(points, np.ndarray) or isinstance(points, np.ndarray) \
        or is_2dlist(points)

    points = np.array(points)
    center = np.mean(points, axis=0)

    def cmp(a, b):
        oa = a - center
        ob = b - center

        # Some corner cases
        if oa[0] >= 0 and ob[0] < 0:
            return 1
        if oa[0] < 0 and ob[0] >= 0:
            return -1

        prod = np.cross(oa, ob)
        if prod > 0:
            return 1
        if prod < 0:
            return -1

        # a, b are on the same line from the center
        return 1 if (oa**2).sum() < (ob**2).sum() else -1

    return sorted(points, key=functools.cmp_to_key(cmp))


def sort_vertex(points_x, points_y):
    # TODO Add typehints & test
    """Sort box vertices in clockwise order from left-top first.

    Args:
        points_x (list[float]): x of four vertices.
        points_y (list[float]): y of four vertices.

    Returns:
        tuple[list[float], list[float]]: Sorted x and y of four vertices.

        - sorted_points_x (list[float]): x of sorted four vertices.
        - sorted_points_y (list[float]): y of sorted four vertices.
    """
    assert is_list_of(points_x, (float, int))
    assert is_list_of(points_y, (float, int))
    assert len(points_x) == 4
    assert len(points_y) == 4
    vertices = np.stack((points_x, points_y), axis=-1).astype(np.float32)
    vertices = _sort_vertex(vertices)
    sorted_points_x = list(vertices[:, 0])
    sorted_points_y = list(vertices[:, 1])
    return sorted_points_x, sorted_points_y


def _sort_vertex(vertices):
    # TODO Add typehints & docstring & test
    assert vertices.ndim == 2
    assert vertices.shape[-1] == 2
    N = vertices.shape[0]
    if N == 0:
        return vertices

    center = np.mean(vertices, axis=0)
    directions = vertices - center
    angles = np.arctan2(directions[:, 1], directions[:, 0])
    sort_idx = np.argsort(angles)
    vertices = vertices[sort_idx]

    left_top = np.min(vertices, axis=0)
    dists = np.linalg.norm(left_top - vertices, axis=-1, ord=2)
    lefttop_idx = np.argmin(dists)
    indexes = (np.arange(N, dtype=np.int_) + lefttop_idx) % N
    return vertices[indexes]


def sort_vertex8(points):
    # TODO Add typehints & docstring & test
    """Sort vertex with 8 points [x1 y1 x2 y2 x3 y3 x4 y4]"""
    assert len(points) == 8
    vertices = _sort_vertex(np.array(points, dtype=np.float32).reshape(-1, 2))
    sorted_box = list(vertices.flatten())
    return sorted_box

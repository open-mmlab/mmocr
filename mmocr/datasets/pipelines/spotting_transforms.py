import numpy as np
from mmdet.core import PolygonMasks
from mmdet.datasets.builder import PIPELINES
from scipy.special import comb as n_over_k

from mmocr.utils import check_argument


@PIPELINES.register_module()
class GenerateBezierPoints:
    """Generate bezier curve annotation for ABCNet: Real-time Scene Text
    Spotting with Adaptive Bezier-Curve Network.

    [https://arxiv.org/abs/2002.10200].
    """

    def __init__(self):
        ...

    def bezier_coeff(self, ts):

        def mtk(n, t, k):
            return t**k * (1 - t)**(n - k) * n_over_k(n, k)

        return [[mtk(3, t, k) for k in range(4)] for t in ts]

    def bezier_fit(self, xs, ys):
        dy = ys[1:] - ys[:-1]
        dx = xs[1:] - xs[:-1]
        dt = (dx**2 + dy**2)**0.5
        t = dt / dt.sum()
        t = np.hstack(([0], t))
        t = t.cumsum()

        data = np.column_stack((xs, ys))
        pseudo_inv = np.linalg.pinv(self.bezier_coeff(t))

        control_points = pseudo_inv.dot(data)
        return control_points[1:-1]

    def __call__(self, results):
        polygon_masks = results['gt_masks'].masks
        assert check_argument.is_2dlist(polygon_masks)

        bezier_points = []
        for poly in polygon_masks:
            poly = poly[0]
            assert len(poly) % 4 == 0
            num_points = int(len(poly) / 2)
            num_side_points = int(len(poly) / 4)

            curve_data_top = poly[0:num_points].reshape(num_side_points, 2)
            curve_data_bottom = poly[num_points:].reshape(num_side_points, 2)

            xs_top = curve_data_top[:, 0]
            ys_top = curve_data_top[:, 1]
            control_points_top = self.bezier_fit(xs_top, ys_top)

            xs_bottom = curve_data_bottom[:, 0]
            ys_bottom = curve_data_bottom[:, 1]
            control_points_bottom = self.bezier_fit(xs_bottom, ys_bottom)

            control_points = np.vstack([
                curve_data_top[0:1], control_points_top, curve_data_top[-1:],
                curve_data_bottom[0:1], control_points_bottom,
                curve_data_bottom[-1:]
            ]).flatten()
            bezier_points.append([control_points])
        bezier_points = PolygonMasks(bezier_points, results['img_shape'][0],
                                     results['img_shape'][1])

        results['control_points'] = bezier_points
        return results

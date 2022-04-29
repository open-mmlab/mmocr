import numpy as np
from mmdet.core import PolygonMasks

import mmocr.datasets.pipelines.spotting_transforms as spotting_transforms


def test_gen_bezier_points():
    bezier_generator = spotting_transforms.GenerateBezierPoints()

    results = {}
    results['img_shape'] = (495, 478, 3)
    text_polys = [[
        np.array([
            117.937904, 105.95588, 155.4281, 89.77941, 192.91829, 79.2647,
            230.4085, 75.22059, 268.67972, 82.5, 306.16992, 92.20588,
            344.44116, 104.338234, 331.1634, 190.07353, 299.92157, 178.75,
            269.4608, 170.66176, 238.21895, 166.61765, 207.75816, 169.04411,
            176.51634, 172.2794, 146.05556, 187.64706
        ])
    ]]
    results['gt_masks'] = PolygonMasks(text_polys, 495, 478)

    target = np.array([
        117.93790436, 105.95587921, 188.15441215, 61.44344027, 271.94136158,
        74.6403483, 344.44116211, 104.33823395, 331.16339111, 190.0735321,
        273.90236857, 165.7008312, 201.93595295, 153.79687672, 146.05555725,
        187.64706421
    ])

    output = bezier_generator(results)['control_points'].masks[0][0]
    assert np.allclose(output, target)

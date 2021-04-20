import numpy as np

from mmocr.core import end2end_show_result


def test_end2end_show_result():
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    end2end_res = {
        'result': [{
            'box': [51, 88, 51, 62, 85, 62, 85, 88],
            'box_score': 0.9417,
            'text': 'hell',
            'text_score': 0.8834
        }]
    }

    vis_img = end2end_show_result(img, end2end_res)
    assert vis_img.shape[0] == 100
    assert vis_img.shape[1] == 200
    assert vis_img.shape[2] == 3

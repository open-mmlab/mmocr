# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmocr.utils import check_argument


class BasePostprocessor:

    def __init__(
        self,
        text_repr_type='poly',
    ):
        assert text_repr_type in ['poly', 'quad'
                                  ], f'Invalid text repr type {text_repr_type}'

        self.text_repr_type = text_repr_type

    def is_valid_instance(self, area, confidence, area_thresh,
                          confidence_thresh):

        return bool(area >= area_thresh and confidence > confidence_thresh)

    def resize_boundary(self, boundaries, scale_factor):
        """Rescale boundaries via scale_factor.

        Args:
            boundaries (list[list[float]]): The boundary list. Each boundary
                has :math:`2k+1` elements with :math:`k>=4`.
            scale_factor (ndarray): The scale factor of size :math:`(4,)`.

        Returns:
            list[list[float]]: The scaled boundaries.
        """
        assert check_argument.is_2dlist(boundaries)
        assert isinstance(scale_factor, np.ndarray)
        assert scale_factor.shape[0] == 4

        for b in boundaries:
            sz = len(b)
            check_argument.valid_boundary(b, True)
            b[:sz -
              1] = (np.array(b[:sz - 1]) *
                    (np.tile(scale_factor[:2], int(
                        (sz - 1) / 2)).reshape(1, sz - 1))).flatten().tolist()
        return boundaries

    def merge_text_spotter_result(self, boundary_results, recog_results,
                                  img_metas):
        """merge detection results and recognition results.

        Args:
            boundary_results (list[list[list[float]]]): The boundary list. Each
            boundary has :math:`2k+1` elements with :math:`k>=4`.
            recog_results (list[list[dict]]): The recognition list. the dict
            must contain 'text' and 'score', which is the format of recognizer
            output.

        Returns:
            list[dict]: end-to-end text recognition results of
            text detection and text recognition
            [
                {
                    "filename": "img_xxx.jpg"
                    "result":
                        [{
                            "box": [159, 82, 488, 428 ...],
                            "box_score":"0.620622",
                            "text":"horse123",
                            "text_score": "0.88"}
                        ],
                }
            ]
        """
        assert len(boundary_results) == len(recog_results) == len(img_metas)
        text_spotter_results = list()
        for boundary_result, recog_result, img_meta in zip(
                boundary_results, recog_results, img_metas):
            single_img_res = dict()
            single_img_res['filename'] = img_meta['filename']
            single_img_res['result'] = list()
            for boundary, recog in zip(boundary_result, recog_result):
                single_img_res['result'].append(
                    dict(
                        box=boundary[:-1],
                        box_score=boundary[-1],
                        text=recog['text'],
                        text_score=recog['score']))
            text_spotter_results.append(single_img_res)
        return text_spotter_results

    def __call__(self, preds):
        pass

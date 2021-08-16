# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv

from mmocr.core import imshow_pred_boundary, seg2boundary


class TextDetectorMixin:
    """The class for implementing text detector auxiliary methods."""

    def __init__(self, show_score):
        self.show_score = show_score

    def get_boundary(self, results):
        """Convert segmentation into text boundaries.

        Args:
           results (tuple): The result tuple. The first element is
               segmentation while the second is its scores.

        Returns:
           results (dict): A result dict containing 'boundary_result'.
        """

        assert isinstance(results, tuple)

        instance_num = len(results[1][0])
        boundaries = []
        for i in range(instance_num):
            seg = results[1][0][i]
            score = results[0][0][i][-1]
            boundary = seg2boundary(seg, self.text_repr_type, score)
            if boundary is not None:
                boundaries.append(boundary)

        results = dict(boundary_result=boundaries)
        return results

    def show_result(self,
                    img,
                    result,
                    score_thr=0.5,
                    bbox_color='green',
                    text_color='green',
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (dict): The results to draw over `img`.
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.imshow_pred_boundary`
        """
        img = mmcv.imread(img)
        img = img.copy()
        boundaries = None
        labels = None
        if 'boundary_result' in result.keys():
            boundaries = result['boundary_result']
            labels = [0] * len(boundaries)

        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        if boundaries is not None:
            imshow_pred_boundary(
                img,
                boundaries,
                labels,
                score_thr=score_thr,
                boundary_color=bbox_color,
                text_color=text_color,
                thickness=thickness,
                font_scale=font_scale,
                win_name=win_name,
                show=show,
                wait_time=wait_time,
                out_file=out_file,
                show_score=self.show_score)

        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, '
                          'result image will be returned')
        return img

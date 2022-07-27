# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import mmcv
import numpy as np
from mmengine import Visualizer

from mmocr.registry import VISUALIZERS
from mmocr.structures import TextDetDataSample


@VISUALIZERS.register_module()
class TextDetLocalVisualizer(Visualizer):
    """The MMOCR Text Detection Local Visualizer.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): The origin image to draw. The format
            should be RGB. Defaults to None.
        with_poly (bool): Whether to draw polygons. Defaults to True.
        with_bbox (bool): Whether to draw bboxes. Defaults to False.
        vis_backends (list, optional): Visual backend config list.
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
        gt_color (Union[str, tuple, list[str], list[tuple]]): The
            colors of GT polygons and bboxes. ``colors`` can have the same
            length with lines or just single value. If ``colors`` is single
            value, all the lines will have the same colors. Refer to
            `matplotlib.colors` for full list of formats that are accepted.
            Defaults to 'g'.
        pred_color (Union[str, tuple, list[str], list[tuple]]): The
            colors of pred polygons and bboxes. ``colors`` can have the same
            length with lines or just single value. If ``colors`` is single
            value, all the lines will have the same colors. Refer to
            `matplotlib.colors` for full list of formats that are accepted.
            Defaults to 'r'.
        line_width (int, float): The linewidth of lines. Defaults to 2.
        alpha (float): The transparency of bboxes or polygons. Defaults to 0.8.
    """

    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 with_poly: bool = True,
                 with_bbox: bool = False,
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 gt_color: Union[str, Tuple, List[str], List[Tuple]] = 'g',
                 pred_color: Union[str, Tuple, List[str], List[Tuple]] = 'r',
                 line_width: Union[int, float] = 2,
                 alpha: float = 0.8) -> None:
        super().__init__(
            name=name,
            image=image,
            vis_backends=vis_backends,
            save_dir=save_dir)
        self.with_poly = with_poly
        self.with_bbox = with_bbox
        self.gt_color = gt_color
        self.pred_color = pred_color
        self.line_width = line_width
        self.alpha = alpha

    def add_datasample(self,
                       name: str,
                       image: np.ndarray,
                       gt_sample: Optional['TextDetDataSample'] = None,
                       pred_sample: Optional['TextDetDataSample'] = None,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       show: bool = False,
                       wait_time: int = 0,
                       out_file: Optional[str] = None,
                       pred_score_thr: float = 0.3,
                       step: int = 0) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be
        saved to ``out_file``. This is usually used when the display
        is not available.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to draw.
            gt_sample (:obj:`TextDetDataSample`, optional): GT
                TextDetDataSample. Defaults to None.
            pred_sample (:obj:`TextDetDataSample`, optional): Predicted
                TextDetDataSample. Defaults to None.
            draw_gt (bool): Whether to draw GT TextDetDataSample.
                Defaults to True.
            draw_pred (bool): Whether to draw Predicted TextDetDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn image. Default to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_file (str): Path to output file. Defaults to None.
            pred_score_thr (float): The threshold to visualize the bboxes
                and masks. Defaults to 0.3.
            step (int): Global step value to record. Defaults to 0.
        """
        gt_img_data = None
        pred_img_data = None

        if draw_gt and gt_sample is not None and 'gt_instances' in gt_sample:
            gt_instances = gt_sample.gt_instances

            self.set_image(image)

            if self.with_poly and 'polygons' in gt_instances:
                gt_polygons = gt_instances.polygons
                gt_polygons = [
                    gt_polygon.reshape(-1, 2) for gt_polygon in gt_polygons
                ]
                self.draw_polygons(
                    gt_polygons,
                    alpha=self.alpha,
                    edge_colors=self.gt_color,
                    line_widths=self.line_width)

            if self.with_bbox and 'bboxes' in gt_instances:
                gt_bboxes = gt_instances.bboxes
                self.draw_bboxes(
                    gt_bboxes,
                    alpha=self.alpha,
                    edge_colors=self.gt_color,
                    line_widths=self.line_width)

            gt_img_data = self.get_image()

        if draw_pred and pred_sample is not None \
                and 'pred_instances' in pred_sample:
            pred_instances = pred_sample.pred_instances
            pred_instances = pred_instances[
                pred_instances.scores > pred_score_thr].cpu()

            self.set_image(image)

            if self.with_poly and 'polygons' in pred_instances:
                pred_polygons = pred_instances.polygons
                pred_polygons = [
                    pred_polygon.reshape(-1, 2)
                    for pred_polygon in pred_polygons
                ]
                self.draw_polygons(
                    pred_polygons,
                    alpha=self.alpha,
                    edge_colors=self.pred_color,
                    line_widths=self.line_width)

            if self.with_bbox and 'bboxes' in pred_instances:
                pred_bboxes = pred_instances.bboxes
                self.draw_bboxes(
                    pred_bboxes,
                    alpha=self.alpha,
                    edge_colors=self.pred_color,
                    line_widths=self.line_width)

            pred_img_data = self.get_image()

        if gt_img_data is not None and pred_img_data is not None:
            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
        elif gt_img_data is not None:
            drawn_img = gt_img_data
        else:
            drawn_img = pred_img_data

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)
        else:
            self.add_image(name, drawn_img, step)

        if out_file is not None:
            mmcv.imwrite(drawn_img[..., ::-1], out_file)

# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
import torch

from mmocr.registry import VISUALIZERS
from mmocr.structures import TextDetDataSample
from .base_visualizer import BaseLocalVisualizer


@VISUALIZERS.register_module()
class TextDetLocalVisualizer(BaseLocalVisualizer):
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
        gt_ignored_color (Union[str, tuple, list[str], list[tuple]]): The
            colors of ignored GT polygons and bboxes. ``colors`` can have
            the same length with lines or just single value. If ``colors``
            is single value, all the lines will have the same colors. Refer
            to `matplotlib.colors` for full list of formats that are accepted.
            Defaults to 'b'.
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
                 gt_ignored_color: Union[str, Tuple, List[str],
                                         List[Tuple]] = 'b',
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
        self.gt_ignored_color = gt_ignored_color
        self.pred_color = pred_color
        self.line_width = line_width
        self.alpha = alpha

    def _draw_instances(
        self,
        image: np.ndarray,
        bboxes: Union[np.ndarray, torch.Tensor],
        polygons: Sequence[np.ndarray],
        color: Union[str, Tuple, List[str], List[Tuple]] = 'g',
    ) -> np.ndarray:
        """Draw bboxes and polygons on image.

        Args:
            image (np.ndarray): The origin image to draw.
            bboxes (Union[np.ndarray, torch.Tensor]): The bboxes to draw.
            polygons (Sequence[np.ndarray]): The polygons to draw.
            color (Union[str, tuple, list[str], list[tuple]]): The
                colors of polygons and bboxes. ``colors`` can have the same
                length with lines or just single value. If ``colors`` is
                single value, all the lines will have the same colors. Refer
                to `matplotlib.colors` for full list of formats that are
                accepted. Defaults to 'g'.

        Returns:
            np.ndarray: The image with bboxes and polygons drawn.
        """
        if polygons is not None and self.with_poly:
            polygons = [polygon.reshape(-1, 2) for polygon in polygons]
            image = self.get_polygons_image(
                image, polygons, filling=True, colors=color, alpha=self.alpha)
        if bboxes is not None and self.with_bbox:
            image = self.get_bboxes_image(
                image,
                bboxes,
                colors=color,
                line_width=self.line_width,
                alpha=self.alpha)
        return image

    def add_datasample(self,
                       name: str,
                       image: np.ndarray,
                       data_sample: Optional['TextDetDataSample'] = None,
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
            data_sample (:obj:`TextDetDataSample`, optional):
                TextDetDataSample which contains gt and prediction. Defaults
                    to None.
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
        cat_images = []
        if data_sample is not None:
            if draw_gt and 'gt_instances' in data_sample:
                gt_instances = data_sample.gt_instances
                gt_img_data = image.copy()
                if gt_instances.get('ignored', None) is not None:
                    ignore_flags = gt_instances.ignored
                    gt_ignored_instances = gt_instances[ignore_flags]
                    gt_ignored_polygons = gt_ignored_instances.get(
                        'polygons', None)
                    gt_ignored_bboxes = gt_ignored_instances.get(
                        'bboxes', None)
                    gt_img_data = self._draw_instances(gt_img_data,
                                                       gt_ignored_bboxes,
                                                       gt_ignored_polygons,
                                                       self.gt_ignored_color)
                    gt_instances = gt_instances[~ignore_flags]
                gt_polygons = gt_instances.get('polygons', None)
                gt_bboxes = gt_instances.get('bboxes', None)
                gt_img_data = self._draw_instances(gt_img_data, gt_bboxes,
                                                   gt_polygons, self.gt_color)
                cat_images.append(gt_img_data)
            if draw_pred and 'pred_instances' in data_sample:
                pred_instances = data_sample.pred_instances
                pred_instances = pred_instances[
                    pred_instances.scores > pred_score_thr].cpu()
                pred_polygons = pred_instances.get('polygons', None)
                pred_bboxes = pred_instances.get('bboxes', None)
                pred_img_data = self._draw_instances(image.copy(), pred_bboxes,
                                                     pred_polygons,
                                                     self.pred_color)
                cat_images.append(pred_img_data)
        cat_images = self._cat_image(cat_images, axis=1)
        if cat_images is None:
            cat_images = image
        if show:
            self.show(cat_images, win_name=name, wait_time=wait_time)
        else:
            self.add_image(name, cat_images, step)

        if out_file is not None:
            mmcv.imwrite(cat_images[..., ::-1], out_file)

        self.set_image(cat_images)
        return self.get_image()

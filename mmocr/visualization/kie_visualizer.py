# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, List, Optional, Sequence, Union

import mmcv
import numpy as np
import torch
from matplotlib.collections import PatchCollection
from matplotlib.patches import FancyArrow
from mmengine.visualization import Visualizer
from mmengine.visualization.utils import (check_type, check_type_and_length,
                                          color_val_matplotlib, tensor2ndarray,
                                          value2list)

from mmocr.registry import VISUALIZERS
from mmocr.structures import KIEDataSample
from .base_visualizer import BaseLocalVisualizer


@VISUALIZERS.register_module()
class KIELocalVisualizer(BaseLocalVisualizer):
    """The MMOCR Text Detection Local Visualizer.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to None.
        vis_backends (list, optional): Visual backend config list.
            Default to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
        fig_save_cfg (dict): Keyword parameters of figure for saving.
            Defaults to empty dict.
        fig_show_cfg (dict): Keyword parameters of figure for showing.
            Defaults to empty dict.
        is_openset (bool, optional): Whether the visualizer is used in
            OpenSet. Defaults to False.
    """

    def __init__(self,
                 name='kie_visualizer',
                 is_openset: bool = False,
                 **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.is_openset = is_openset

    def _draw_edge_label(self,
                         image: np.ndarray,
                         edge_labels: Union[np.ndarray, torch.Tensor],
                         bboxes: Union[np.ndarray, torch.Tensor],
                         texts: Sequence[str],
                         arrow_colors: str = 'g') -> np.ndarray:
        """Draw edge labels on image.

        Args:
            image (np.ndarray): The origin image to draw. The format
                should be RGB.
            edge_labels (np.ndarray or torch.Tensor): The edge labels to draw.
                The shape of edge_labels should be (N, N), where N is the
                number of texts.
            bboxes (np.ndarray or torch.Tensor): The bboxes to draw. The shape
                of bboxes should be (N, 4), where N is the number of texts.
            texts (Sequence[str]): The texts to draw. The length of texts
                should be the same as the number of bboxes.
            arrow_colors (str, optional): The colors of arrows. Refer to
                `matplotlib.colors` for full list of formats that are accepted.
                Defaults to 'g'.

        Returns:
            np.ndarray: The image with edge labels drawn.
        """
        pairs = np.where(edge_labels > 0)
        key_bboxes = bboxes[pairs[0]]
        value_bboxes = bboxes[pairs[1]]
        x_data = np.stack([(key_bboxes[:, 2] + key_bboxes[:, 0]) / 2,
                           (value_bboxes[:, 0] + value_bboxes[:, 2]) / 2],
                          axis=-1)
        y_data = np.stack([(key_bboxes[:, 1] + key_bboxes[:, 3]) / 2,
                           (value_bboxes[:, 1] + value_bboxes[:, 3]) / 2],
                          axis=-1)
        key_index = np.array(list(set(pairs[0])))
        val_index = np.array(list(set(pairs[1])))
        key_texts = [texts[i] for i in key_index]
        val_texts = [texts[i] for i in val_index]

        self.set_image(image)
        if key_texts:
            self.draw_texts(
                key_texts, (bboxes[key_index, :2] + bboxes[key_index, 2:]) / 2,
                colors='k',
                horizontal_alignments='center',
                vertical_alignments='center')
        if val_texts:
            self.draw_texts(
                val_texts, (bboxes[val_index, :2] + bboxes[val_index, 2:]) / 2,
                colors='k',
                horizontal_alignments='center',
                vertical_alignments='center')
        self.draw_arrows(
            x_data,
            y_data,
            colors=arrow_colors,
            line_widths=0.3,
            arrow_tail_widths=0.05,
            arrow_head_widths=5,
            overhangs=1,
            arrow_shapes='full')
        return self.get_image()

    def _draw_instances(
        self,
        image: np.ndarray,
        bbox_labels: Union[np.ndarray, torch.Tensor],
        bboxes: Union[np.ndarray, torch.Tensor],
        polygons: Sequence[np.ndarray],
        edge_labels: Union[np.ndarray, torch.Tensor],
        texts: Sequence[str],
        class_names: Dict,
        is_openset: bool = False,
        arrow_colors: str = 'g',
    ) -> np.ndarray:
        """Draw instances on image.

        Args:
            image (np.ndarray): The origin image to draw. The format
                should be RGB.
            bbox_labels (np.ndarray or torch.Tensor): The bbox labels to draw.
                The shape of bbox_labels should be (N,), where N is the
                number of texts.
            bboxes (np.ndarray or torch.Tensor): The bboxes to draw. The shape
                of bboxes should be (N, 4), where N is the number of texts.
            polygons (Sequence[np.ndarray]): The polygons to draw. The length
                of polygons should be the same as the number of bboxes.
            edge_labels (np.ndarray or torch.Tensor): The edge labels to draw.
                The shape of edge_labels should be (N, N), where N is the
                number of texts.
            texts (Sequence[str]): The texts to draw. The length of texts
                should be the same as the number of bboxes.
            class_names (dict): The class names for bbox labels.
            is_openset (bool): Whether the dataset is openset. Defaults to
                False.
            arrow_colors (str, optional): The colors of arrows. Refer to
                `matplotlib.colors` for full list of formats that are accepted.
                Defaults to 'g'.

        Returns:
            np.ndarray: The image with instances drawn.
        """
        img_shape = image.shape[:2]
        empty_shape = (img_shape[0], img_shape[1], 3)

        text_image = np.full(empty_shape, 255, dtype=np.uint8)
        text_image = self.get_labels_image(text_image, texts, bboxes)

        classes_image = np.full(empty_shape, 255, dtype=np.uint8)
        bbox_classes = [class_names[int(i)]['name'] for i in bbox_labels]
        classes_image = self.get_labels_image(classes_image, bbox_classes,
                                              bboxes)
        if polygons:
            polygons = [polygon.reshape(-1, 2) for polygon in polygons]
            image = self.get_polygons_image(
                image, polygons, filling=True, colors=self.PALETTE)
            text_image = self.get_polygons_image(
                text_image, polygons, colors=self.PALETTE)
            classes_image = self.get_polygons_image(
                classes_image, polygons, colors=self.PALETTE)
        else:
            image = self.get_bboxes_image(
                image, bboxes, filling=True, colors=self.PALETTE)
            text_image = self.get_bboxes_image(
                text_image, bboxes, colors=self.PALETTE)
            classes_image = self.get_bboxes_image(
                classes_image, bboxes, colors=self.PALETTE)
        cat_image = [image, text_image, classes_image]
        if is_openset:
            edge_image = np.full(empty_shape, 255, dtype=np.uint8)
            edge_image = self._draw_edge_label(edge_image, edge_labels, bboxes,
                                               texts, arrow_colors)
            cat_image.append(edge_image)
        return self._cat_image(cat_image, axis=1)

    def add_datasample(self,
                       name: str,
                       image: np.ndarray,
                       data_sample: Optional['KIEDataSample'] = None,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       show: bool = False,
                       wait_time: int = 0,
                       pred_score_thr: float = None,
                       out_file: Optional[str] = None,
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
            data_sample (:obj:`KIEDataSample`, optional):
                KIEDataSample which contains gt and prediction. Defaults
                    to None.
            draw_gt (bool): Whether to draw GT KIEDataSample.
                Defaults to True.
            draw_pred (bool): Whether to draw Predicted KIEDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn image. Default to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            pred_score_thr (float): The threshold to visualize the bboxes
                and masks. Defaults to 0.3.
            out_file (str): Path to output file. Defaults to None.
            step (int): Global step value to record. Defaults to 0.
        """
        cat_images = list()

        if draw_gt:
            gt_bboxes = data_sample.gt_instances.bboxes
            gt_labels = data_sample.gt_instances.labels
            gt_texts = data_sample.gt_instances.texts
            gt_polygons = data_sample.gt_instances.get('polygons', None)
            gt_edge_labels = data_sample.gt_instances.get('edge_labels', None)
            gt_img_data = self._draw_instances(image, gt_labels, gt_bboxes,
                                               gt_polygons, gt_edge_labels,
                                               gt_texts,
                                               self.dataset_meta['category'],
                                               self.is_openset, 'g')
            cat_images.append(gt_img_data)
        if draw_pred:
            gt_bboxes = data_sample.gt_instances.bboxes
            pred_labels = data_sample.pred_instances.labels
            gt_texts = data_sample.gt_instances.texts
            gt_polygons = data_sample.gt_instances.get('polygons', None)
            pred_edge_labels = data_sample.pred_instances.get(
                'edge_labels', None)
            pred_img_data = self._draw_instances(image, pred_labels, gt_bboxes,
                                                 gt_polygons, pred_edge_labels,
                                                 gt_texts,
                                                 self.dataset_meta['category'],
                                                 self.is_openset, 'r')
            cat_images.append(pred_img_data)

        cat_images = self._cat_image(cat_images, axis=0)
        if cat_images is None:
            cat_images = image

        if show:
            self.show(cat_images, win_name=name, wait_time=wait_time)
        else:
            self.add_image(name, cat_images, step)

        if out_file is not None:
            mmcv.imwrite(cat_images[..., ::-1], out_file)

    def draw_arrows(self,
                    x_data: Union[np.ndarray, torch.Tensor],
                    y_data: Union[np.ndarray, torch.Tensor],
                    colors: Union[str, tuple, List[str], List[tuple]] = 'C1',
                    line_widths: Union[Union[int, float],
                                       List[Union[int, float]]] = 1,
                    line_styles: Union[str, List[str]] = '-',
                    arrow_tail_widths: Union[Union[int, float],
                                             List[Union[int, float]]] = 0.001,
                    arrow_head_widths: Union[Union[int, float],
                                             List[Union[int, float]]] = None,
                    arrow_head_lengths: Union[Union[int, float],
                                              List[Union[int, float]]] = None,
                    arrow_shapes: Union[str, List[str]] = 'full',
                    overhangs: Union[int, List[int]] = 0) -> 'Visualizer':
        """Draw single or multiple arrows.

        Args:
            x_data (np.ndarray or torch.Tensor): The x coordinate of
                each line' start and end points.
            y_data (np.ndarray, torch.Tensor): The y coordinate of
                each line' start and end points.
            colors (str or tuple or list[str or tuple]): The colors of
                lines. ``colors`` can have the same length with lines or just
                single value. If ``colors`` is single value, all the lines
                will have the same colors. Reference to
                https://matplotlib.org/stable/gallery/color/named_colors.html
                for more details. Defaults to 'g'.
            line_widths (int or float or list[int or float]):
                The linewidth of lines. ``line_widths`` can have
                the same length with lines or just single value.
                If ``line_widths`` is single value, all the lines will
                have the same linewidth. Defaults to 2.
            line_styles (str or list[str]]): The linestyle of lines.
                ``line_styles`` can have the same length with lines or just
                single value. If ``line_styles`` is single value, all the
                lines will have the same linestyle. Defaults to '-'.
            arrow_tail_widths (int or float or list[int, float]):
                The width of arrow tails. ``arrow_tail_widths`` can have
                the same length with lines or just single value. If
                ``arrow_tail_widths`` is single value, all the lines will
                have the same width. Defaults to 0.001.
            arrow_head_widths (int or float or list[int, float]):
                The width of arrow heads. ``arrow_head_widths`` can have
                the same length with lines or just single value. If
                ``arrow_head_widths`` is single value, all the lines will
                have the same width. Defaults to None.
            arrow_head_lengths (int or float or list[int, float]):
                The length of arrow heads. ``arrow_head_lengths`` can have
                the same length with lines or just single value. If
                ``arrow_head_lengths`` is single value, all the lines will
                have the same length. Defaults to None.
            arrow_shapes (str or list[str]]): The shapes of arrow heads.
                ``arrow_shapes`` can have the same length with lines or just
                single value. If ``arrow_shapes`` is single value, all the
                lines will have the same shape. Defaults to 'full'.
            overhangs (int or list[int]]): The overhangs of arrow heads.
                ``overhangs`` can have the same length with lines or just
                single value. If ``overhangs`` is single value, all the lines
                will have the same overhangs. Defaults to 0.
        """
        check_type('x_data', x_data, (np.ndarray, torch.Tensor))
        x_data = tensor2ndarray(x_data)
        check_type('y_data', y_data, (np.ndarray, torch.Tensor))
        y_data = tensor2ndarray(y_data)
        assert x_data.shape == y_data.shape, (
            '`x_data` and `y_data` should have the same shape')
        assert x_data.shape[-1] == 2, (
            f'The shape of `x_data` should be (N, 2), but got {x_data.shape}')
        if len(x_data.shape) == 1:
            x_data = x_data[None]
            y_data = y_data[None]
        number_arrow = x_data.shape[0]
        check_type_and_length('colors', colors, (str, tuple, list),
                              number_arrow)
        colors = value2list(colors, (str, tuple), number_arrow)
        colors = color_val_matplotlib(colors)  # type: ignore
        check_type_and_length('line_widths', line_widths, (int, float),
                              number_arrow)
        line_widths = value2list(line_widths, (int, float), number_arrow)
        check_type_and_length('arrow_tail_widths', arrow_tail_widths,
                              (int, float), number_arrow)
        check_type_and_length('line_styles', line_styles, str, number_arrow)
        line_styles = value2list(line_styles, str, number_arrow)
        arrow_tail_widths = value2list(arrow_tail_widths, (int, float),
                                       number_arrow)
        check_type_and_length('arrow_head_widths', arrow_head_widths,
                              (int, float, type(None)), number_arrow)
        arrow_head_widths = value2list(arrow_head_widths,
                                       (int, float, type(None)), number_arrow)
        check_type_and_length('arrow_head_lengths', arrow_head_lengths,
                              (int, float, type(None)), number_arrow)
        arrow_head_lengths = value2list(arrow_head_lengths,
                                        (int, float, type(None)), number_arrow)
        check_type_and_length('arrow_shapes', arrow_shapes, (str, list),
                              number_arrow)
        arrow_shapes = value2list(arrow_shapes, (str, list), number_arrow)
        check_type('overhang', overhangs, int)
        overhangs = value2list(overhangs, int, number_arrow)

        lines = np.concatenate(
            (x_data.reshape(-1, 2, 1), y_data.reshape(-1, 2, 1)), axis=-1)
        if not self._is_posion_valid(lines):
            warnings.warn(
                'Warning: The line is out of bounds,'
                ' the drawn line may not be in the image', UserWarning)
        arrows = []
        for i in range(number_arrow):
            arrows.append(
                FancyArrow(
                    *tuple(lines[i, 0]),
                    *tuple(lines[i, 1] - lines[i, 0]),
                    linestyle=line_styles[i],
                    color=colors[i],
                    length_includes_head=True,
                    width=arrow_tail_widths[i],
                    head_width=arrow_head_widths[i],
                    head_length=arrow_head_lengths[i],
                    overhang=overhangs[i],
                    shape=arrow_shapes[i],
                    linewidth=line_widths[i]))
        p = PatchCollection(arrows, match_original=True)
        self.ax_save.add_collection(p)
        return self

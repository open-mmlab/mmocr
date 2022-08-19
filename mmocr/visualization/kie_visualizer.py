# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
from typing import Dict, List, Optional, Sequence, Union

import mmcv
import numpy as np
import torch
from matplotlib.collections import PatchCollection
from matplotlib.patches import FancyArrow
from mmengine import Visualizer
from mmengine.visualization.utils import (check_type, check_type_and_length,
                                          color_val_matplotlib, tensor2ndarray,
                                          value2list)

from mmocr.registry import VISUALIZERS
from mmocr.structures import KIEDataSample

PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
           (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192),
           (250, 170, 30), (100, 170, 30), (220, 220, 0), (175, 116, 175),
           (250, 0, 30), (165, 42, 42), (255, 77, 255), (0, 226, 252),
           (182, 182, 255), (0, 82, 0), (120, 166, 157), (110, 76, 0),
           (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
           (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
           (255, 99, 164), (92, 0, 73), (133, 129, 255), (78, 180, 255),
           (0, 228, 0), (174, 255, 243), (45, 89, 255), (134, 134, 103),
           (145, 148, 174), (255, 208, 186), (197, 226, 255), (171, 134, 1),
           (109, 63, 54), (207, 138, 255), (151, 0, 95), (9, 80, 61),
           (84, 105, 51), (74, 65, 105), (166, 196, 102), (208, 195, 210),
           (255, 109, 65), (0, 143, 149), (179, 0, 194), (209, 99, 106),
           (5, 121, 0), (227, 255, 205), (147, 186, 208), (153, 69, 1),
           (3, 95, 161), (163, 255, 0), (119, 0, 170), (0, 182, 199),
           (0, 165, 120), (183, 130, 88), (95, 32, 0), (130, 114, 135),
           (110, 129, 133), (166, 74, 118), (219, 142, 185), (79, 210, 114),
           (178, 90, 62), (65, 70, 15), (127, 167, 115), (59, 105, 106),
           (142, 108, 45), (196, 172, 0), (95, 54, 80), (128, 76, 255),
           (201, 57, 1), (246, 0, 122), (191, 162, 208)]


@VISUALIZERS.register_module()
class KieLocalVisualizer(Visualizer):
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

    @staticmethod
    def _draw_labels(visualizer: Visualizer,
                     image: np.ndarray,
                     labels: Union[np.ndarray, torch.Tensor],
                     bboxes: Union[np.ndarray, torch.Tensor],
                     colors: Union[str, Sequence[str]] = 'k',
                     font_size: Union[int, float] = 10,
                     auto_font_size: bool = False) -> np.ndarray:
        """Draw labels on image.

        Args:
            image (np.ndarray): The origin image to draw. The format
                should be RGB.
            labels (Union[np.ndarray, torch.Tensor]): The labels to draw.
            bboxes (Union[np.ndarray, torch.Tensor]): The bboxes to draw.
            colors (Union[str, Sequence[str]]): The colors of labels.
                ``colors`` can have the same length with labels or just single
                value. If ``colors`` is single value, all the labels will have
                the same colors. Refer to `matplotlib.colors` for full list of
                formats that are accepted. Defaults to 'k'.
            font_size (Union[int, float]): The font size of labels. Defaults
                to 10.
            auto_font_size (bool): Whether to automatically adjust font size.
                Defaults to False.
        """
        if colors is not None and isinstance(colors, Sequence):
            size = math.ceil(len(labels) / len(colors))
            colors = (colors * size)[:len(labels)]
        if auto_font_size:
            assert font_size is not None and isinstance(
                font_size, (int, float))
            font_size = (bboxes[:, 2:] - bboxes[:, :2]).min(-1) * font_size
            font_size = font_size.tolist()
        visualizer.set_image(image)
        visualizer.draw_texts(
            labels, (bboxes[:, :2] + bboxes[:, 2:]) / 2,
            vertical_alignments='center',
            horizontal_alignments='center',
            colors='k',
            font_sizes=font_size)
        return visualizer.get_image()

    @staticmethod
    def _draw_polygons(visualizer: Visualizer,
                       image: np.ndarray,
                       polygons: Sequence[np.ndarray],
                       colors: Union[str, Sequence[str]] = 'g',
                       filling: bool = False,
                       line_width: Union[int, float] = 0.5,
                       alpha: float = 0.5) -> np.ndarray:
        if colors is not None and isinstance(colors, Sequence):
            size = math.ceil(len(polygons) / len(colors))
            colors = (colors * size)[:len(polygons)]
        visualizer.set_image(image)
        if filling:
            visualizer.draw_polygons(
                polygons,
                face_colors=colors,
                edge_colors=colors,
                line_widths=line_width,
                alpha=alpha)
        else:
            visualizer.draw_polygons(
                polygons,
                edge_colors=colors,
                line_widths=line_width,
                alpha=alpha)
        return visualizer.get_image()

    @staticmethod
    def _draw_bboxes(visualizer: Visualizer,
                     image: np.ndarray,
                     bboxes: Union[np.ndarray, torch.Tensor],
                     colors: Union[str, Sequence[str]] = 'g',
                     filling: bool = False,
                     line_width: Union[int, float] = 0.5,
                     alpha: float = 0.5) -> np.ndarray:
        if colors is not None and isinstance(colors, Sequence):
            size = math.ceil(len(bboxes) / len(colors))
            colors = (colors * size)[:len(bboxes)]
        visualizer.set_image(image)
        if filling:
            visualizer.draw_bboxes(
                bboxes,
                face_colors=colors,
                edge_colors=colors,
                line_widths=line_width,
                alpha=alpha)
        else:
            visualizer.draw_bboxes(
                bboxes,
                edge_colors=colors,
                line_widths=line_width,
                alpha=alpha)
        return visualizer.get_image()

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
        edge_labels (np.ndarray, torch.Tensor): The edge labels to draw.
            The shape of edge_labels should be (N, N), where N is the
            number of texts.
        bboxes (np.ndarray, torch.Tensor): The bboxes to draw. The shape of
            bboxes should be (N, 4), where N is the number of texts.
        texts (Sequence[str]): The texts to draw. The length of texts
            should be the same as the number of bboxes.
        arrow_colors (str, optional): The colors of arrows. Refer to
            `matplotlib.colors` for full list of formats that are accepted.
            Defaults to 'g'.
        """
        pairs = np.where(edge_labels > 0)
        key_bboxes = bboxes[pairs[0]]
        value_bboxes = bboxes[pairs[1]]
        x_datas = np.stack([(key_bboxes[:, 2] + key_bboxes[:, 0]) / 2,
                            (value_bboxes[:, 0] + value_bboxes[:, 2]) / 2],
                           axis=-1)
        y_datas = np.stack([(key_bboxes[:, 1] + key_bboxes[:, 3]) / 2,
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
            x_datas,
            y_datas,
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
            bbox_labels (np.ndarray, torch.Tensor): The bbox labels to draw.
                The shape of bbox_labels should be (N,), where N is the
                number of texts.
            bboxes (np.ndarray, torch.Tensor): The bboxes to draw. The shape of
                bboxes should be (N, 4), where N is the number of texts.
            polygons (Sequence[np.ndarray]): The polygons to draw. The length
                of polygons should be the same as the number of bboxes.
            edge_labels (np.ndarray, torch.Tensor): The edge labels to draw.
                The shape of edge_labels should be (N, N), where N is the
                number of texts.
            texts (Sequence[str]): The texts to draw. The length of texts
                should be the same as the number of bboxes.
            class_names (dict): The class names for bbox labels.
            is_openset (bool): Whether the dataset is openset. Default: False.
        """
        img_shape = image.shape[:2]
        empty_shape = (img_shape[0], img_shape[1], 3)

        if polygons:
            polygons = [polygon.reshape(-1, 2) for polygon in polygons]
        if polygons:
            image = self._draw_polygons(
                self, image, polygons, filling=True, colors=PALETTE)
        else:
            image = self._draw_bboxes(
                self, image, bboxes, filling=True, colors=PALETTE)

        text_image = np.full(empty_shape, 255, dtype=np.uint8)
        text_image = self._draw_labels(self, text_image, texts, bboxes)
        if polygons:
            text_image = self._draw_polygons(
                self, text_image, polygons, colors=PALETTE)
        else:
            text_image = self._draw_bboxes(
                self, text_image, bboxes, colors=PALETTE)

        classes_image = np.full(empty_shape, 255, dtype=np.uint8)
        bbox_classes = [class_names[int(i)]['name'] for i in bbox_labels]
        classes_image = self._draw_labels(self, classes_image, bbox_classes,
                                          bboxes)
        if polygons:
            classes_image = self._draw_polygons(
                self, classes_image, polygons, colors=PALETTE)
        else:
            classes_image = self._draw_bboxes(
                self, classes_image, bboxes, colors=PALETTE)

        edge_image = None
        if is_openset:
            edge_image = np.full(empty_shape, 255, dtype=np.uint8)
            edge_image = self._draw_edge_label(edge_image, edge_labels, bboxes,
                                               texts, arrow_colors)
        cat_image = []
        for i in [image, text_image, classes_image, edge_image]:
            if i is not None:
                cat_image.append(i)
        return np.concatenate(cat_image, axis=1)

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
        gt_img_data = None
        pred_img_data = None

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
        if gt_img_data is not None and pred_img_data is not None:
            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=0)
        elif gt_img_data is not None:
            drawn_img = gt_img_data
        elif pred_img_data is not None:
            drawn_img = pred_img_data
        else:
            drawn_img = image

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)
        else:
            self.add_image(name, drawn_img, step)

        if out_file is not None:
            mmcv.imwrite(drawn_img[..., ::-1], out_file)

    def draw_arrows(self,
                    x_datas: Union[np.ndarray, torch.Tensor],
                    y_datas: Union[np.ndarray, torch.Tensor],
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
            x_datas (np.ndarray or torch.Tensor): The x coordinate of
                each line' start and end points.
            y_datas (np.ndarray, torch.Tensor): The y coordinate of
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
        check_type('x_datas', x_datas, (np.ndarray, torch.Tensor))
        x_datas = tensor2ndarray(x_datas)
        check_type('y_datas', y_datas, (np.ndarray, torch.Tensor))
        y_datas = tensor2ndarray(y_datas)
        assert x_datas.shape == y_datas.shape, (
            '`x_datas` and `y_datas` should have the same shape')
        assert x_datas.shape[-1] == 2, (
            f'The shape of `x_datas` should be (N, 2), but got {x_datas.shape}'
        )
        if len(x_datas.shape) == 1:
            x_datas = x_datas[None]
            y_datas = y_datas[None]
        number_arrow = x_datas.shape[0]
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
            (x_datas.reshape(-1, 2, 1), y_datas.reshape(-1, 2, 1)), axis=-1)
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

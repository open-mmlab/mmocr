# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Sequence, Union

import numpy as np
import torch
from mmengine.visualization import Visualizer

from mmocr.registry import VISUALIZERS


@VISUALIZERS.register_module()
class BaseLocalVisualizer(Visualizer):
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
    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
               (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
               (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
               (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
               (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
               (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
               (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
               (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
               (134, 134, 103), (145, 148, 174), (255, 208, 186),
               (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
               (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
               (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
               (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
               (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
               (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
               (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
               (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
               (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
               (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
               (191, 162, 208)]

    def get_labels_image(self,
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
        if colors is not None and isinstance(colors, (list, tuple)):
            size = math.ceil(len(labels) / len(colors))
            colors = (colors * size)[:len(labels)]
        if auto_font_size:
            assert font_size is not None and isinstance(
                font_size, (int, float))
            font_size = (bboxes[:, 2:] - bboxes[:, :2]).min(-1) * font_size
            font_size = font_size.tolist()
        self.set_image(image)
        self.draw_texts(
            labels, (bboxes[:, :2] + bboxes[:, 2:]) / 2,
            vertical_alignments='center',
            horizontal_alignments='center',
            colors='k',
            font_sizes=font_size)
        return self.get_image()

    def get_polygons_image(self,
                           image: np.ndarray,
                           polygons: Sequence[np.ndarray],
                           colors: Union[str, Sequence[str]] = 'g',
                           filling: bool = False,
                           line_width: Union[int, float] = 0.5,
                           alpha: float = 0.5) -> np.ndarray:
        """Draw polygons on image.

        Args:
            image (np.ndarray): The origin image to draw. The format
                should be RGB.
            polygons (Sequence[np.ndarray]): The polygons to draw. The shape
                should be (N, 2).
            colors (Union[str, Sequence[str]]): The colors of polygons.
                ``colors`` can have the same length with polygons or just
                single value. If ``colors`` is single value, all the polygons
                will have the same colors. Refer to `matplotlib.colors` for
                full list of formats that are accepted. Defaults to 'g'.
            filling (bool): Whether to fill the polygons. Defaults to False.
            line_width (Union[int, float]): The line width of polygons.
                Defaults to 0.5.
            alpha (float): The alpha of polygons. Defaults to 0.5.

        Returns:
            np.ndarray: The image with polygons drawn.
        """
        if colors is not None and isinstance(colors, (list, tuple)):
            size = math.ceil(len(polygons) / len(colors))
            colors = (colors * size)[:len(polygons)]
        self.set_image(image)
        if filling:
            self.draw_polygons(
                polygons,
                face_colors=colors,
                edge_colors=colors,
                line_widths=line_width,
                alpha=alpha)
        else:
            self.draw_polygons(
                polygons,
                edge_colors=colors,
                line_widths=line_width,
                alpha=alpha)
        return self.get_image()

    def get_bboxes_image(self: Visualizer,
                         image: np.ndarray,
                         bboxes: Union[np.ndarray, torch.Tensor],
                         colors: Union[str, Sequence[str]] = 'g',
                         filling: bool = False,
                         line_width: Union[int, float] = 0.5,
                         alpha: float = 0.5) -> np.ndarray:
        """Draw bboxes on image.

        Args:
            image (np.ndarray): The origin image to draw. The format
                should be RGB.
            bboxes (Union[np.ndarray, torch.Tensor]): The bboxes to draw.
            colors (Union[str, Sequence[str]]): The colors of bboxes.
                ``colors`` can have the same length with bboxes or just single
                value. If ``colors`` is single value, all the bboxes will have
                the same colors. Refer to `matplotlib.colors` for full list of
                formats that are accepted. Defaults to 'g'.
            filling (bool): Whether to fill the bboxes. Defaults to False.
            line_width (Union[int, float]): The line width of bboxes.
                Defaults to 0.5.
            alpha (float): The alpha of bboxes. Defaults to 0.5.

        Returns:
            np.ndarray: The image with bboxes drawn.
        """
        if colors is not None and isinstance(colors, (list, tuple)):
            size = math.ceil(len(bboxes) / len(colors))
            colors = (colors * size)[:len(bboxes)]
        self.set_image(image)
        if filling:
            self.draw_bboxes(
                bboxes,
                face_colors=colors,
                edge_colors=colors,
                line_widths=line_width,
                alpha=alpha)
        else:
            self.draw_bboxes(
                bboxes,
                edge_colors=colors,
                line_widths=line_width,
                alpha=alpha)
        return self.get_image()

    def _draw_instances(self) -> np.ndarray:
        raise NotImplementedError

    def _cat_image(self, imgs: Sequence[np.ndarray], axis: int) -> np.ndarray:
        """Concatenate images.

        Args:
            imgs (Sequence[np.ndarray]): The images to concatenate.
            axis (int): The axis to concatenate.

        Returns:
            np.ndarray: The concatenated image.
        """
        cat_image = list()
        for img in imgs:
            if img is not None:
                cat_image.append(img)
        if len(cat_image):
            return np.concatenate(cat_image, axis=axis)
        else:
            return None

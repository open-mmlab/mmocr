# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple, Union

import cv2
import mmcv
import numpy as np

from mmocr.registry import VISUALIZERS
from mmocr.structures import TextRecogDataSample
from .base_visualizer import BaseLocalVisualizer


@VISUALIZERS.register_module()
class TextRecogLocalVisualizer(BaseLocalVisualizer):
    """MMOCR Text Detection Local Visualizer.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): The origin image to draw. The format
            should be RGB. Defaults to None.
        vis_backends (list, optional): Visual backend config list.
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
        gt_color (str or tuple[int, int, int]): Colors of GT text. The tuple of
            color should be in RGB order. Or using an abbreviation of color,
            such as `'g'` for `'green'`. Defaults to 'g'.
        pred_color (str or tuple[int, int, int]): Colors of Predicted text.
            The tuple of color should be in RGB order. Or using an abbreviation
            of color, such as `'r'` for `'red'`. Defaults to 'r'.
    """

    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 gt_color: Optional[Union[str, Tuple[int, int, int]]] = 'g',
                 pred_color: Optional[Union[str, Tuple[int, int, int]]] = 'r',
                 **kwargs) -> None:
        super().__init__(
            name=name,
            image=image,
            vis_backends=vis_backends,
            save_dir=save_dir,
            **kwargs)
        self.gt_color = gt_color
        self.pred_color = pred_color

    def _draw_instances(self, image: np.ndarray, text: str) -> np.ndarray:
        """Draw text on image.

        Args:
            image (np.ndarray): The image to draw.
            text (str): The text to draw.

        Returns:
            np.ndarray: The image with text drawn.
        """
        height, width = image.shape[:2]
        empty_img = np.full_like(image, 255)
        self.set_image(empty_img)
        font_size = min(0.5 * width / (len(text) + 1), 0.5 * height)
        self.draw_texts(
            text,
            np.array([width / 2, height / 2]),
            colors=self.gt_color,
            font_sizes=font_size,
            vertical_alignments='center',
            horizontal_alignments='center',
            font_families=self.font_families)
        text_image = self.get_image()
        return text_image

    def add_datasample(self,
                       name: str,
                       image: np.ndarray,
                       data_sample: Optional['TextRecogDataSample'] = None,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       show: bool = False,
                       wait_time: int = 0,
                       pred_score_thr: float = None,
                       out_file: Optional[str] = None,
                       step=0) -> None:
        """Visualize datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be
        saved to ``out_file``. This is usually used when the display
        is not available.

        Args:
            name (str): The image title. Defaults to 'image'.
            image (np.ndarray): The image to draw.
            data_sample (:obj:`TextRecogDataSample`, optional):
                TextRecogDataSample which contains gt and prediction.
                Defaults to None.
            draw_gt (bool): Whether to draw GT TextRecogDataSample.
                Defaults to True.
            draw_pred (bool): Whether to draw Predicted TextRecogDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn image. Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_file (str): Path to output file. Defaults to None.
            step (int): Global step value to record. Defaults to 0.
            pred_score_thr (float): Threshold of prediction score. It's not
                used in this function. Defaults to None.
        """
        height, width = image.shape[:2]
        resize_height = 64
        resize_width = int(1.0 * width / height * resize_height)
        image = cv2.resize(image, (resize_width, resize_height))

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        cat_images = [image]
        if (draw_gt and data_sample is not None and 'gt_text' in data_sample
                and 'item' in data_sample.gt_text):
            gt_text = data_sample.gt_text.item
            cat_images.append(self._draw_instances(image, gt_text))
        if (draw_pred and data_sample is not None
                and 'pred_text' in data_sample
                and 'item' in data_sample.pred_text):
            pred_text = data_sample.pred_text.item
            cat_images.append(self._draw_instances(image, pred_text))
        cat_images = self._cat_image(cat_images, axis=0)

        if show:
            self.show(cat_images, win_name=name, wait_time=wait_time)
        else:
            self.add_image(name, cat_images, step)

        if out_file is not None:
            mmcv.imwrite(cat_images[..., ::-1], out_file)

        self.set_image(cat_images)
        return self.get_image()

# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple, Union

import cv2
import mmcv
import numpy as np
from mmengine import Visualizer

from mmocr.structures import TextRecogDataSample
from mmocr.registry import VISUALIZERS


@VISUALIZERS.register_module()
class TextRecogLocalVisualizer(Visualizer):
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

    def __init__(
            self,
            name: str = 'visualizer',
            image: Optional[np.ndarray] = None,
            vis_backends: Optional[Dict] = None,
            save_dir: Optional[str] = None,
            gt_color: Optional[Union[str, Tuple[int, int, int]]] = 'g',
            pred_color: Optional[Union[str, Tuple[int, int,
                                                  int]]] = 'r') -> None:
        super().__init__(
            name=name,
            image=image,
            vis_backends=vis_backends,
            save_dir=save_dir)
        self.gt_color = gt_color
        self.pred_color = pred_color

    def add_datasample(self,
                       name: str,
                       image: np.ndarray,
                       gt_sample: Optional['TextRecogDataSample'] = None,
                       pred_sample: Optional['TextRecogDataSample'] = None,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       show: bool = False,
                       wait_time: int = 0,
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
            gt_sample (:obj:`TextRecogDataSample`, optional): GT
                TextRecogDataSample. Defaults to None.
            pred_sample (:obj:`TextRecogDataSample`, optional): Predicted
                TextRecogDataSample. Defaults to None.
            draw_gt (bool): Whether to draw GT TextRecogDataSample.
                Defaults to True.
            draw_pred (bool): Whether to draw Predicted TextRecogDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn image. Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_file (str): Path to output file. Defaults to None.
            step (int): Global step value to record. Defaults to 0.
        """
        gt_img_data = None
        pred_img_data = None
        height, width = image.shape[:2]
        resize_height = 64
        resize_width = int(1.0 * width / height * resize_height)
        image = cv2.resize(image, (resize_width, resize_height))
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if draw_gt and gt_sample is not None and 'gt_text' in gt_sample:
            gt_text = gt_sample.gt_text.item
            empty_img = np.full_like(image, 255)
            self.set_image(empty_img)
            font_size = 0.5 * resize_width / len(gt_text)
            self.draw_texts(
                gt_text,
                np.array([resize_width / 2, resize_height / 2]),
                colors=self.gt_color,
                font_sizes=font_size,
                vertical_alignments='center',
                horizontal_alignments='center')
            gt_text_image = self.get_image()
            gt_img_data = np.concatenate((image, gt_text_image), axis=0)

        if (draw_pred and pred_sample is not None
                and 'pred_text' in pred_sample):
            pred_text = pred_sample.pred_text.item
            empty_img = np.full_like(image, 255)
            self.set_image(empty_img)
            font_size = 0.5 * resize_width / len(pred_text)
            self.draw_texts(
                pred_text,
                np.array([resize_width / 2, resize_height / 2]),
                colors=self.pred_color,
                font_sizes=font_size,
                vertical_alignments='center',
                horizontal_alignments='center')
            pred_text_image = self.get_image()
            pred_img_data = np.concatenate((image, pred_text_image), axis=0)

        if gt_img_data is not None and pred_img_data is not None:
            drawn_img = np.concatenate((gt_img_data, pred_text_image), axis=0)
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

# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import mmcv
import numpy as np
import torch
from mmdet.visualization.palette import _get_adaptive_scales
from mmengine.structures import LabelData

from mmocr.registry import VISUALIZERS
from mmocr.visualization.base_visualizer import BaseLocalVisualizer
from projects.LayoutLMv3.structures import SERDataSample


@VISUALIZERS.register_module()
class SERLocalVisualizer(BaseLocalVisualizer):
    """The MMOCR Semantic Entity Recognition Local Visualizer.

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
        bbox_color (Union[str, tuple, list[str], list[tuple]]): The
            colors of bboxes. ``colors`` can have the same
            length with lines or just single value. If ``colors`` is single
            value, all the lines will have the same colors. Refer to
            `matplotlib.colors` for full list of formats that are accepted.
            Defaults to 'b'.
        label_color (Union[str, tuple, list[str], list[tuple]]): The
            colors of gt/pred label. ``colors`` can have
            the same length with lines or just single value. If ``colors``
            is single value, all the lines will have the same colors. Refer
            to `matplotlib.colors` for full list of formats that are accepted.
            Defaults to 'g'.
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
                 bbox_color: Union[str, Tuple, List[str], List[Tuple]] = 'b',
                 label_color: Union[str, Tuple, List[str], List[Tuple]] = 'g',
                 line_width: Union[int, float] = 2,
                 alpha: float = 0.8) -> None:
        super().__init__(
            name=name,
            image=image,
            vis_backends=vis_backends,
            save_dir=save_dir)
        self.with_poly = with_poly
        self.with_bbox = with_bbox
        self.bbox_color = bbox_color
        self.label_color = label_color
        self.line_width = line_width
        self.alpha = alpha

    def _draw_instances(self, image: np.ndarray, bboxes: Union[np.ndarray,
                                                               torch.Tensor],
                        word_ids: List[int], gt_labels: Optional[LabelData],
                        pred_labels: Optional[LabelData]) -> np.ndarray:
        """Draw bboxes and polygons on image.

        Args:
            image (np.ndarray): The origin image to draw.
            bboxes (Union[np.ndarray, torch.Tensor]): The bboxes to draw.
            word_ids (List[int]): The word id of tokens.
            gt_labels (Optional[LabelData]): The gt LabelData.
            pred_labels (Optional[LabelData]): The pred LabelData.
        Returns:
            np.ndarray: The image with bboxes and gt/pred labels drawn.
        """
        # draw bboxes
        if bboxes is not None and self.with_bbox:
            image = self.get_bboxes_image(
                image,
                bboxes,
                colors=self.bbox_color,
                line_width=self.line_width,
                alpha=self.alpha)

        # draw gt/pred labels
        if gt_labels is not None and pred_labels is not None:
            gt_tokens_biolabel = gt_labels.item
            gt_words_label = []
            pred_tokens_biolabel = pred_labels.item
            pred_words_label = []

            if 'score' in pred_labels:
                pred_tokens_biolabel_score = pred_labels.score
                pred_words_label_score = []
            else:
                pred_tokens_biolabel_score = None
                pred_words_label_score = None

            pre_word_id = None
            for idx, cur_word_id in enumerate(word_ids):
                if cur_word_id is not None:
                    if cur_word_id != pre_word_id:
                        gt_words_label_name = gt_tokens_biolabel[idx][2:] \
                            if gt_tokens_biolabel[idx] != 'O' else 'other'
                        gt_words_label.append(gt_words_label_name)
                        pred_words_label_name = pred_tokens_biolabel[idx][2:] \
                            if pred_tokens_biolabel[idx] != 'O' else 'other'
                        pred_words_label.append(pred_words_label_name)
                        if pred_tokens_biolabel_score is not None:
                            pred_words_label_score.append(
                                pred_tokens_biolabel_score[idx])
                pre_word_id = cur_word_id
            assert len(gt_words_label) == len(bboxes)
            assert len(pred_words_label) == len(bboxes)

            areas = (bboxes[:, 3] - bboxes[:, 1]) * (
                bboxes[:, 2] - bboxes[:, 0])
            scales = _get_adaptive_scales(areas)
            positions = bboxes[:, :2] - self.line_width

            self.set_image(image)
            for i, (pos, gt, pred) in enumerate(
                    zip(positions, gt_words_label, pred_words_label)):
                if pred_words_label_score is not None:
                    score = round(float(pred_words_label_score[i]) * 100, 1)
                    label_text = f'{gt} | {pred}({score})'
                else:
                    label_text = f'{gt} | {pred}'

                self.draw_texts(
                    label_text,
                    pos,
                    color=self.label_color if gt == pred else 'r',
                    font_sizes=int(13 * scales[i]))

        return self.get_image()

    def add_datasample(self,
                       name: str,
                       image: np.ndarray,
                       data_sample: Optional[SERDataSample] = None,
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
            bboxes = np.array(data_sample.instances.get('boxes', None))
            # here need to flatten truncation_word_ids
            word_ids = [
                word_id for word_ids in data_sample.truncation_word_ids
                for word_id in word_ids[1:-1]
            ]
            gt_label = data_sample.gt_label if \
                draw_gt and 'gt_label' in data_sample else None
            pred_label = data_sample.pred_label if \
                draw_pred and 'pred_label' in data_sample else None
            draw_img = self._draw_instances(image.copy(), bboxes, word_ids,
                                            gt_label, pred_label)
            cat_images.append(draw_img)
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

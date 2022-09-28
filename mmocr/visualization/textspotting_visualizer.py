# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

import mmcv
import numpy as np
import torch

from mmocr.registry import VISUALIZERS
from mmocr.structures import TextDetDataSample
from mmocr.utils.polygon_utils import poly2bbox
from .base_visualizer import BaseLocalVisualizer


@VISUALIZERS.register_module()
class TextSpottingLocalVisualizer(BaseLocalVisualizer):

    def _draw_instances(
        self,
        image: np.ndarray,
        bboxes: Union[np.ndarray, torch.Tensor],
        polygons: Sequence[np.ndarray],
        texts: Sequence[str],
    ) -> np.ndarray:
        """Draw instances on image.

        Args:
            image (np.ndarray): The origin image to draw. The format
                should be RGB.
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
        text_image = np.full(empty_shape, 255, dtype=np.uint8)
        text_image = self.get_labels_image(
            text_image, labels=texts, bboxes=bboxes)
        if polygons:
            polygons = [polygon.reshape(-1, 2) for polygon in polygons]
            image = self.get_polygons_image(
                image, polygons, filling=True, colors=self.PALETTE)
            text_image = self.get_polygons_image(
                text_image, polygons, colors=self.PALETTE)
        else:
            image = self.get_bboxes_image(
                image, bboxes, filling=True, colors=self.PALETTE)
            text_image = self.get_bboxes_image(
                text_image, bboxes, colors=self.PALETTE)
        return np.concatenate([image, text_image], axis=1)

    def add_datasample(self,
                       name: str,
                       image: np.ndarray,
                       data_sample: Optional['TextDetDataSample'] = None,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       show: bool = False,
                       wait_time: int = 0,
                       pred_score_thr: float = 0.5,
                       out_file: Optional[str] = None,
                       step: int = 0) -> None:
        cat_images = []

        if draw_gt:
            gt_bboxes = data_sample.gt_instances.get('bboxes', None)
            gt_texts = data_sample.gt_instances.texts
            gt_polygons = data_sample.gt_instances.get('polygons', None)
            gt_img_data = self._draw_instances(image, gt_bboxes, gt_polygons,
                                               gt_texts)
            cat_images.append(gt_img_data)

        if draw_pred:
            pred_instances = data_sample.pred_instances
            pred_instances = pred_instances[
                pred_instances.scores > pred_score_thr].cpu().numpy()
            pred_bboxes = pred_instances.get('bboxes', None)
            pred_texts = pred_instances.texts
            pred_polygons = pred_instances.get('polygons', None)
            if pred_bboxes is None:
                pred_bboxes = [poly2bbox(poly) for poly in pred_polygons]
                pred_bboxes = np.array(pred_bboxes)
            pred_img_data = self._draw_instances(image, pred_bboxes,
                                                 pred_polygons, pred_texts)
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

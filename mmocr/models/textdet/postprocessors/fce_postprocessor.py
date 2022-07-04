# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Sequence

import cv2
import numpy as np
import torch
from mmengine import InstanceData
from numpy.fft import ifft

from mmocr.core import TextDetDataSample
from mmocr.registry import MODELS
from mmocr.utils import fill_hole
from .base_postprocessor import BaseTextDetPostProcessor


@MODELS.register_module()
class FCEPostprocessor(BaseTextDetPostProcessor):
    """Decoding predictions of FCENet to instances.

    Args:
        fourier_degree (int): The maximum Fourier transform degree k.
        num_reconstr_points (int): The points number of the polygon
            reconstructed from predicted Fourier coefficients.
        rescale_fields (list[str]): The bbox/polygon field names to
            be rescaled. If None, no rescaling will be performed. Defaults to
            ['polygons'].
        scales (list[int]) : The down-sample scale of each layer. Defaults
            to [8, 16, 32].
        text_repr_type (str): Boundary encoding type 'poly' or 'quad'. Defaults
            to 'poly'.
         alpha (float): The parameter to calculate final scores
            :math:`Score_{final} = (Score_{text region} ^ alpha)
            * (Score_{text center_region}^ beta)`. Defaults to 1.0.
        beta (float): The parameter to calculate final score. Defaults to 2.0.
        score_thr (float): The threshold used to filter out the final
            candidates.Defaults to 0.3.
        nms_thr (float): The threshold of nms. Defaults to 0.1.
    """

    def __init__(self,
                 fourier_degree: int,
                 num_reconstr_points: int,
                 rescale_fields: Sequence[str] = ['polygons'],
                 scales: Sequence[int] = [8, 16, 32],
                 text_repr_type: str = 'poly',
                 alpha: float = 1.0,
                 beta: float = 2.0,
                 score_thr: float = 0.3,
                 nms_thr: float = 0.1,
                 **kwargs) -> None:
        super().__init__(
            text_repr_type=text_repr_type,
            rescale_fields=rescale_fields,
            **kwargs)
        self.fourier_degree = fourier_degree
        self.num_reconstr_points = num_reconstr_points
        self.scales = scales
        self.alpha = alpha
        self.beta = beta
        self.score_thr = score_thr
        self.nms_thr = nms_thr

    def split_results(self, pred_results: List[Dict]) -> List[List[Dict]]:
        """Split batched elements in pred_results along the first dimension
        into ``batch_num`` sub-elements and regather them into a list of dicts.

        Args:
            pred_results (list[dict]): A list of dict with keys of ``cls_res``,
                ``reg_res`` corresponding to the classification result and
                regression result computed from the input tensor with the
                same index. They have the shapes of :math:`(N, C_{cls,i},
                H_i, W_i)` and :math:`(N, C_{out,i}, H_i, W_i)`.

        Returns:
            list[list[dict]]: N lists. Each list contains three dicts from
            different feature level.
        """
        assert isinstance(pred_results, list) and len(pred_results) == len(
            self.scales)

        fields = list(pred_results[0].keys())
        batch_num = len(pred_results[0][fields[0]])
        level_num = len(pred_results)
        results = []
        for i in range(batch_num):
            batch_list = []
            for level in range(level_num):
                feat_dict = {}
                for field in fields:
                    feat_dict[field] = pred_results[level][field][i]
                batch_list.append(feat_dict)
            results.append(batch_list)
        return results

    def get_text_instances(self, pred_results: Sequence[Dict],
                           data_sample: TextDetDataSample
                           ) -> TextDetDataSample:
        """Get text instance predictions of one image.

        Args:
            pred_results (List[dict]): A list of dict with keys of ``cls_res``,
                ``reg_res`` corresponding to the classification result and
                regression result computed from the input tensor with the
                same index. They have the shapes of :math:`(N, C_{cls,i}, H_i,
                W_i)` and :math:`(N, C_{out,i}, H_i, W_i)`.
            data_sample (TextDetDataSample): Datasample of an image.

        Returns:
            TextDetDataSample: A new DataSample with predictions filled in.
            Polygons and results are saved in
            ``TextDetDataSample.pred_instances.polygons``. The confidence
            scores are saved in ``TextDetDataSample.pred_instances.scores``.
        """
        assert len(pred_results) == len(self.scales)
        data_sample.pred_instances = InstanceData()
        data_sample.pred_instances.polygons = []
        data_sample.pred_instances.scores = []

        result_polys = []
        result_scores = []
        for idx, pred_result in enumerate(pred_results):
            # TODO: Scale can be calculated given image shape and feature
            # shape. This param can be removed in the future.
            polygons, scores = self._get_text_instances_single(
                pred_result, self.scales[idx])
            result_polys += polygons
            result_scores += scores
        result_polys, result_scores = self.poly_nms(result_polys,
                                                    result_scores,
                                                    self.nms_thr)
        for result_poly, result_score in zip(result_polys, result_scores):
            result_poly = np.array(result_poly, dtype=np.float32)
            data_sample.pred_instances.polygons.append(result_poly)
            data_sample.pred_instances.scores.append(result_score)
        data_sample.pred_instances.scores = torch.FloatTensor(
            data_sample.pred_instances.scores)

        return data_sample

    def _get_text_instances_single(self, pred_result: Dict, scale: int):
        """Get text instance predictions from one feature level.

        Args:
            pred_result (dict): A dict with keys of ``cls_res``, ``reg_res``
                corresponding to the classification result and regression
                result computed from the input tensor with the same index.
                They have the shapes of :math:`(1, C_{cls,i}, H_i, W_i)` and
                :math:`(1, C_{out,i}, H_i, W_i)`.
            scale (int): Scale of current feature map which equals to
                img_size / feat_size.

        Returns:
            result_polys (list[ndarray]): A list of polygons after postprocess.
            result_scores (list[ndarray]): A list of scores after postprocess.
        """

        cls_pred = pred_result['cls_res']
        tr_pred = cls_pred[0:2].softmax(dim=0).data.cpu().numpy()
        tcl_pred = cls_pred[2:].softmax(dim=0).data.cpu().numpy()

        reg_pred = pred_result['reg_res'].permute(1, 2, 0).data.cpu().numpy()
        x_pred = reg_pred[:, :, :2 * self.fourier_degree + 1]
        y_pred = reg_pred[:, :, 2 * self.fourier_degree + 1:]

        score_pred = (tr_pred[1]**self.alpha) * (tcl_pred[1]**self.beta)
        tr_pred_mask = (score_pred) > self.score_thr
        tr_mask = fill_hole(tr_pred_mask)

        tr_contours, _ = cv2.findContours(
            tr_mask.astype(np.uint8), cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE)  # opencv4

        mask = np.zeros_like(tr_mask)

        result_polys = []
        result_scores = []
        for cont in tr_contours:
            deal_map = mask.copy().astype(np.int8)
            cv2.drawContours(deal_map, [cont], -1, 1, -1)

            score_map = score_pred * deal_map
            score_mask = score_map > 0
            xy_text = np.argwhere(score_mask)
            dxy = xy_text[:, 1] + xy_text[:, 0] * 1j

            x, y = x_pred[score_mask], y_pred[score_mask]
            c = x + y * 1j
            c[:, self.fourier_degree] = c[:, self.fourier_degree] + dxy
            c *= scale

            polygons = self._fourier2poly(c, self.num_reconstr_points)
            scores = score_map[score_mask].reshape(-1, 1).tolist()
            polygons, scores = self.poly_nms(polygons, scores, self.nms_thr)
            result_polys += polygons
            result_scores += scores

        result_polys, result_scores = self.poly_nms(result_polys,
                                                    result_scores,
                                                    self.nms_thr)

        if self.text_repr_type == 'quad':
            new_polys = []
            for poly in result_polys:
                poly = np.array(poly).reshape(-1, 2).astype(np.float32)
                points = cv2.boxPoints(cv2.minAreaRect(poly))
                points = np.int0(points)
                new_polys.append(points.reshape(-1))

            return new_polys, result_scores
        return result_polys, result_scores

    def _fourier2poly(self,
                      fourier_coeff: np.ndarray,
                      num_reconstr_points: int = 50):
        """ Inverse Fourier transform
            Args:
                fourier_coeff (ndarray): Fourier coefficients shaped (n, 2k+1),
                    with n and k being candidates number and Fourier degree
                    respectively.
                num_reconstr_points (int): Number of reconstructed polygon
                    points. Defaults to 50.

            Returns:
                List[ndarray]: The reconstructed polygons.
            """

        a = np.zeros((len(fourier_coeff), num_reconstr_points),
                     dtype='complex')
        k = (len(fourier_coeff[0]) - 1) // 2

        a[:, 0:k + 1] = fourier_coeff[:, k:]
        a[:, -k:] = fourier_coeff[:, :k]

        poly_complex = ifft(a) * num_reconstr_points
        polygon = np.zeros((len(fourier_coeff), num_reconstr_points, 2))
        polygon[:, :, 0] = poly_complex.real
        polygon[:, :, 1] = poly_complex.imag
        return polygon.astype('int32').reshape(
            (len(fourier_coeff), -1)).tolist()

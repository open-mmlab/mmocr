# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
from torch import Tensor

from mmocr.structures import TextDetDataSample
from mmocr.utils import boundary_iou, rescale_polygons


class BaseTextDetPostProcessor:
    """Base postprocessor for text detection models.

    Args:
        text_repr_type (str): The boundary encoding type, 'poly' or 'quad'.
            Defaults to 'poly'.
        rescale_fields (list[str], optional): The bbox/polygon field names to
            be rescaled. If None, no rescaling will be performed.
        train_cfg (dict, optional): The parameters to be passed to
            ``self.get_text_instances`` in training. Defaults to None.
        test_cfg (dict, optional): The parameters to be passed to
            ``self.get_text_instances`` in testing. Defaults to None.
    """

    def __init__(self,
                 text_repr_type: str = 'poly',
                 rescale_fields: Optional[Sequence[str]] = None,
                 train_cfg: Optional[Dict] = None,
                 test_cfg: Optional[Dict] = None) -> None:
        assert text_repr_type in ['poly', 'quad']
        assert rescale_fields is None or isinstance(rescale_fields, list)
        assert train_cfg is None or isinstance(train_cfg, dict)
        assert test_cfg is None or isinstance(test_cfg, dict)
        self.text_repr_type = text_repr_type
        self.rescale_fields = rescale_fields
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def __call__(self,
                 pred_results: Union[Tensor, List[Tensor]],
                 data_samples: Sequence[TextDetDataSample],
                 training: bool = False) -> Sequence[TextDetDataSample]:
        """Postprocess pred_results according to metainfos in data_samples.

        Args:
            pred_results (Union[Tensor, List[Tensor]]): The prediction results
                stored in a tensor or a list of tensor. Usually each item to
                be post-processed is expected to be a batched tensor.
            data_samples (list[TextDetDataSample]): Batch of data_samples,
                each corresponding to a prediction result.
            training (bool): Whether the model is in training mode. Defaults to
                False.

        Returns:
            list[TextDetDataSample]: Batch of post-processed datasamples.
        """
        cfg = self.train_cfg if training else self.test_cfg
        if cfg is None:
            cfg = {}
        pred_results = self.split_results(pred_results)
        process_single = partial(self._process_single, **cfg)
        results = list(map(process_single, pred_results, data_samples))

        return results

    def _process_single(self, pred_result: Union[Tensor, List[Tensor]],
                        data_sample: TextDetDataSample,
                        **kwargs) -> TextDetDataSample:
        """Process prediction results from one image.

        Args:
            pred_result (Union[Tensor, List[Tensor]]): Prediction results of an
                image.
            data_sample (TextDetDataSample): Datasample of an image.
        """

        results = self.get_text_instances(pred_result, data_sample, **kwargs)

        if self.rescale_fields and len(self.rescale_fields) > 0:
            assert isinstance(self.rescale_fields, list)
            assert set(self.rescale_fields).issubset(
                set(results.pred_instances.keys()))
            results = self.rescale(results, data_sample.scale_factor)
        return results

    def rescale(self, results: TextDetDataSample,
                scale_factor: Sequence[int]) -> TextDetDataSample:
        """Rescale results in ``results.pred_instances`` according to
        ``scale_factor``, whose keys are defined in ``self.rescale_fields``.
        Usually used to rescale bboxes and/or polygons.

        Args:
            results (TextDetDataSample): The post-processed prediction results.
            scale_factor (tuple(int)): (w_scale, h_scale)

        Returns:
            TextDetDataSample: Prediction results with rescaled results.
        """
        scale_factor = np.asarray(scale_factor)
        for key in self.rescale_fields:
            results.pred_instances[key] = rescale_polygons(
                results.pred_instances[key], scale_factor, mode='div')
        return results

    def get_text_instances(self, pred_results: Union[Tensor, List[Tensor]],
                           data_sample: TextDetDataSample,
                           **kwargs) -> TextDetDataSample:
        """Get text instance predictions of one image.

        Args:
            pred_result (tuple(Tensor)): Prediction results of an image.
            data_sample (TextDetDataSample): Datasample of an image.
            **kwargs: Other parameters. Configurable via ``__init__.train_cfg``
                and ``__init__.test_cfg``.

        Returns:
            TextDetDataSample: A new DataSample with predictions filled in.
            The polygon/bbox results are usually saved in
            ``TextDetDataSample.pred_instances.polygons`` or
            ``TextDetDataSample.pred_instances.bboxes``. The confidence scores
            are saved in ``TextDetDataSample.pred_instances.scores``.
        """
        raise NotImplementedError

    def split_results(
        self, pred_results: Union[Tensor, List[Tensor]]
    ) -> Union[List[Tensor], List[List[Tensor]]]:
        """Split batched tensor(s) along the first dimension pack split tensors
        into a list.

        Args:
            pred_results (tensor or list[tensor]): Raw result tensor(s) from
                detection head. Each tensor usually has the shape of (N, ...)

        Returns:
            list[tensor] or list[list[tensor]]: N tensors if ``pred_results``
                is a tensor, or a list of N lists of tensors if
                ``pred_results`` is a list of tensors.
        """
        assert isinstance(pred_results, Tensor) or mmengine.is_seq_of(
            pred_results, Tensor)

        if mmengine.is_seq_of(pred_results, Tensor):
            for i in range(1, len(pred_results)):
                assert pred_results[0].shape[0] == pred_results[i].shape[0], \
                    'The first dimension of all tensors should be the same'

        batch_num = len(pred_results) if isinstance(pred_results, Tensor) else\
            len(pred_results[0])
        results = []
        for i in range(batch_num):
            if isinstance(pred_results, Tensor):
                results.append(pred_results[i])
            else:
                results.append([])
                for tensor in pred_results:
                    results[i].append(tensor[i])
        return results

    def poly_nms(self, polygons: List[np.ndarray], scores: List[float],
                 threshold: float) -> Tuple[List[np.ndarray], List[float]]:
        """Non-maximum suppression for text detection.

        Args:
            polygons (list[ndarray]): List of polygons.
            scores (list[float]): List of scores.
            threshold (float): Threshold for NMS.

        Returns:
            tuple(keep_polys, keep_scores):

            - keep_polys (list[ndarray]): List of preserved polygons after NMS.
            - keep_scores (list[float]): List of preserved scores after NMS.
        """
        assert isinstance(polygons, list)
        assert isinstance(scores, list)
        assert len(polygons) == len(scores)

        polygons = [
            np.hstack((polygon, score))
            for polygon, score in zip(polygons, scores)
        ]
        polygons = np.array(sorted(polygons, key=lambda x: x[-1]))
        keep_polys = []
        keep_scores = []
        index = [i for i in range(len(polygons))]

        while len(index) > 0:
            keep_polys.append(polygons[index[-1]][:-1].tolist())
            keep_scores.append(polygons[index[-1]][-1])
            A = polygons[index[-1]][:-1]
            index = np.delete(index, -1)

            iou_list = np.zeros((len(index), ))
            for i in range(len(index)):
                B = polygons[index[i]][:-1]

                iou_list[i] = boundary_iou(A, B, 1)
            remove_index = np.where(iou_list > threshold)
            index = np.delete(index, remove_index)

        return keep_polys, keep_scores

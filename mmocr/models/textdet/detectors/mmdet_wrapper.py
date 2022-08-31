# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import cv2
import torch
from mmdet.structures import DetDataSample
from mmdet.structures import SampleList as MMDET_SampleList
from mmdet.structures.mask import bitmap_to_polygon
from mmengine.model import BaseModel
from mmengine.structures import InstanceData

from mmocr.registry import MODELS
from mmocr.utils.bbox_utils import bbox2poly
from mmocr.utils.typing import DetSampleList

ForwardResults = Union[Dict[str, torch.Tensor], List[DetDataSample],
                       Tuple[torch.Tensor], torch.Tensor]


@MODELS.register_module()
class MMDetWrapper(BaseModel):
    """A wrapper of MMDet's model.

    Args:
        cfg (dict): The config of the model.
        text_repr_type (str): The boundary encoding type 'poly' or 'quad'.
            Defaults to 'poly'.
    """

    def __init__(self, cfg: Dict, text_repr_type: str = 'poly') -> None:
        data_preprocessor = cfg.pop('data_preprocessor')
        data_preprocessor.update(_scope_='mmdet')
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=None)
        cfg['_scope_'] = 'mmdet'
        self.wrapped_model = MODELS.build(cfg)
        self.text_repr_type = text_repr_type

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[Union[DetSampleList,
                                             MMDET_SampleList]] = None,
                mode: str = 'tensor',
                **kwargs) -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method works in three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`DetDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle either back propagation or
        parameter update, which are supposed to be done in :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (list[:obj:`DetDataSample`] or
                list[:obj:`TextDetDataSample`]): The annotation data of every
                sample. When in "predict" mode, it should be a list of
                :obj:`TextDetDataSample`. Otherwise they are
                :obj:`DetDataSample`s. Defaults to None.
            mode (str): Running mode. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`TextDetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'predict':
            ocr_data_samples = data_samples
            data_samples = []
            for i in range(len(ocr_data_samples)):
                data_samples.append(
                    DetDataSample(metainfo=ocr_data_samples[i].metainfo))

        results = self.wrapped_model.forward(inputs, data_samples, mode,
                                             **kwargs)

        if mode == 'predict':
            results = self.adapt_predictions(results, ocr_data_samples)

        return results

    def adapt_predictions(self, data: MMDET_SampleList,
                          data_samples: DetSampleList) -> DetSampleList:
        """Convert Instance datas from MMDet into MMOCR's format.

        Args:
            data: (list[DetDataSample]): Detection results of the
                input images. Each DetDataSample usually contain
                'pred_instances'. And the ``pred_instances`` usually
                contains following keys.
                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor, Optional): Has a shape (num_instances, H, W).
            data_samples (list[:obj:`TextDetDataSample`]): The annotation data
                of every samples.

        Returns:
            list[TextDetDataSample]: A list of N datasamples containing ground
                truth and prediction results.
                The polygon results are saved in
                ``TextDetDataSample.pred_instances.polygons``
                The confidence scores are saved in
                ``TextDetDataSample.pred_instances.scores``.
        """
        for i, det_data_sample in enumerate(data):
            data_samples[i].pred_instances = InstanceData()
            # convert mask to polygons if mask exists
            if 'masks' in det_data_sample.pred_instances.keys():
                masks = det_data_sample.pred_instances.masks.cpu().numpy()
                polygons = []
                scores = []
                for mask_idx, mask in enumerate(masks):
                    contours, _ = bitmap_to_polygon(mask)
                    polygons += [contour.reshape(-1) for contour in contours]
                    scores += [
                        det_data_sample.pred_instances.scores[mask_idx].cpu()
                    ] * len(contours)
                # filter invalid polygons
                filterd_polygons = []
                keep_idx = []
                for poly_idx, polygon in enumerate(polygons):
                    if len(polygon) < 6:
                        continue
                    filterd_polygons.append(polygon)
                    keep_idx.append(poly_idx)
                # convert by text_repr_type
                if self.text_repr_type == 'quad':
                    for i, poly in enumerate(filterd_polygons):
                        rect = cv2.minAreaRect(poly)
                        vertices = cv2.boxPoints(rect)
                        poly = vertices.flatten()
                        filterd_polygons[i] = poly

                data_samples[i].pred_instances.polygons = filterd_polygons
                data_samples[i].pred_instances.scores = torch.FloatTensor(
                    scores)[keep_idx]
            else:
                bboxes = det_data_sample.pred_instances.bboxes.cpu().numpy()
                polygons = [bbox2poly(bbox) for bbox in bboxes]
                data_samples[i].pred_instances.polygons = polygons
                data_samples[i].pred_instances.scores = torch.FloatTensor(
                    det_data_sample.pred_instances.scores.cpu())

        return data_samples

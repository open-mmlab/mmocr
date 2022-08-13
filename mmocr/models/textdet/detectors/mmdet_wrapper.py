# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple, Union

import cv2
import torch
from mmdet.structures import DetDataSample, OptSampleList
from mmdet.structures.mask import bitmap_to_polygon
from mmengine import InstanceData
from mmengine.model import BaseModel

from mmocr.registry import MODELS
from mmocr.structures import TextDetDataSample
from mmocr.utils.bbox_utils import bbox2poly

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
                batch_inputs: torch.Tensor,
                batch_data_samples: OptSampleList = None,
                mode: str = 'tensor',
                **kwargs) -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`DetDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            batch_inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            batch_data_samples (list[:obj:`DetDataSample`], optional): The
                annotation data of every samples. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`TextDetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        results = self.wrapped_model.forward(batch_inputs, batch_data_samples,
                                             mode, **kwargs)
        if mode == 'predict':
            results = self.adapt_predictions(results)

        return results

    def adapt_predictions(self, data: List[DetDataSample]
                          ) -> List[TextDetDataSample]:
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

        Returns:
            list[TextDetDataSample]: A list of N datasamples of prediction
                results.
                The polygon results are saved in
                ``TextDetDataSample.pred_instances.polygons``
                The confidence scores are saved in
                ``TextDetDataSample.pred_instances.scores``.
        """
        results = []
        for data_sample in data:
            result = TextDetDataSample()
            result.pred_instances = InstanceData()
            # convert mask to polygons if mask exists
            if 'masks' in data_sample.pred_instances.keys():
                masks = data_sample.pred_instances.masks.cpu().numpy()
                polygons = []
                scores = []
                for mask_idx, mask in enumerate(masks):
                    contours, _ = bitmap_to_polygon(mask)
                    polygons += [contour.reshape(-1) for contour in contours]
                    scores += [
                        data_sample.pred_instances.scores[mask_idx].cpu()
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

                result.pred_instances.polygons = filterd_polygons
                result.pred_instances.scores = torch.FloatTensor(
                    scores)[keep_idx]
            else:
                bboxes = data_sample.pred_instances.bboxes.cpu().numpy()
                polygons = [bbox2poly(bbox) for bbox in bboxes]
                result.pred_instances.polygons = polygons
                result.pred_instances.scores = torch.FloatTensor(
                    data_sample.pred_instances.scores.cpu())
            results.append(result)

        return results

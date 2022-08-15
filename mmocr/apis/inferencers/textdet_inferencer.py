# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

from mmocr.structures import TextDetDataSample
from .base_inferencer import BaseInferencer


class TextDetInferencer(BaseInferencer):

    def _pred2dict(self, data_sample: TextDetDataSample) -> Dict:
        """Extract elements necessary to represent a prediction into a
        dictionary. It's better to contain only basic data elements such as
        strings and numbers in order to guarantee it's json-serializable.

        Args:
            data_sample (TextDetDataSample): The data sample to be converted.

        Returns:
            dict: The output dictionary.
        """
        result = {}
        pred_instances = data_sample.pred_instances
        result['polygons'] = []
        for polygon in pred_instances.polygons:
            result['polygons'].append(polygon.tolist())
        result['scores'] = pred_instances.scores.numpy().tolist()
        return result

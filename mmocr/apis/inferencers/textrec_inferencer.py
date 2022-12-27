# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import numpy as np

from mmocr.structures import TextRecogDataSample
from .base_mmocr_inferencer import BaseMMOCRInferencer


class TextRecInferencer(BaseMMOCRInferencer):

    def pred2dict(self, data_sample: TextRecogDataSample) -> Dict:
        """Extract elements necessary to represent a prediction into a
        dictionary. It's better to contain only basic data elements such as
        strings and numbers in order to guarantee it's json-serializable.

        Args:
            data_sample (TextRecogDataSample): The data sample to be converted.

        Returns:
            dict: The output dictionary.
        """
        result = {}
        result['text'] = data_sample.pred_text.item
        result['scores'] = float(np.mean(data_sample.pred_text.score))
        return result

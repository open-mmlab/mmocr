# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import numpy as np

from mmocr.structures import TextRecogDataSample
from .base_mmocr_inferencer import BaseMMOCRInferencer


class TextRecInferencer(BaseMMOCRInferencer):
    """Text Recognition inferencer.

    Args:
        model (str, optional): Path to the config file or the model name
            defined in metafile. For example, it could be
            "crnn_mini-vgg_5e_mj" or
            "configs/textrecog/crnn/crnn_mini-vgg_5e_mj.py".
            If model is not specified, user must provide the
            `weights` saved by MMEngine which contains the config string.
            Defaults to None.
        weights (str, optional): Path to the checkpoint. If it is not specified
            and model is a model name of metafile, the weights will be loaded
            from metafile. Defaults to None.
        device (str, optional): Device to run inference. If None, the available
            device will be automatically used. Defaults to None.
        scope (str, optional): The scope of the model. Defaults to "mmocr".
    """

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
        score = self._array2list(data_sample.pred_text.score)
        result['scores'] = float(np.mean(score))
        return result

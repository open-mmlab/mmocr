# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import numpy as np
from mmengine.model import BaseTTAModel

from mmocr.registry import MODELS
from mmocr.utils.typing_utils import RecSampleList


@MODELS.register_module()
class EncoderDecoderRecognizerTTAModel(BaseTTAModel):
    """Merge augmented recognition results. It will select the best result
    according average scores from all augmented results.

    Examples:
        >>> tta_model = dict(
        >>>     type='EncoderDecoderRecognizerTTAModel')
        >>>
        >>> tta_pipeline = [
        >>>     dict(
        >>>         type='LoadImageFromFile',
        >>>         color_type='grayscale',
        >>>         file_client_args=file_client_args),
        >>>     dict(
        >>>         type='TestTimeAug',
        >>>         transforms=[
        >>>             [
        >>>                 dict(
        >>>                     type='ConditionApply',
        >>>                     true_transforms=[
        >>>                         dict(
        >>>                             type='ImgAugWrapper',
        >>>                             args=[dict(cls='Rot90', k=0, keep_size=False)]) # noqa: E501
        >>>                     ],
        >>>                     condition="results['img_shape'][1]<results['img_shape'][0]" # noqa: E501
        >>>                 ),
        >>>                 dict(
        >>>                     type='ConditionApply',
        >>>                     true_transforms=[
        >>>                         dict(
        >>>                             type='ImgAugWrapper',
        >>>                             args=[dict(cls='Rot90', k=1, keep_size=False)]) # noqa: E501
        >>>                     ],
        >>>                     condition="results['img_shape'][1]<results['img_shape'][0]" # noqa: E501
        >>>                 ),
        >>>                 dict(
        >>>                     type='ConditionApply',
        >>>                     true_transforms=[
        >>>                         dict(
        >>>                             type='ImgAugWrapper',
        >>>                             args=[dict(cls='Rot90', k=3, keep_size=False)])
        >>>                     ],
        >>>                     condition="results['img_shape'][1]<results['img_shape'][0]"
        >>>                 ),
        >>>             ],
        >>>             [
        >>>                 dict(
        >>>                     type='RescaleToHeight',
        >>>                     height=32,
        >>>                     min_width=32,
        >>>                     max_width=None,
        >>>                     width_divisor=16)
        >>>             ],
        >>>             # add loading annotation after ``Resize`` because ground truth
        >>>             # does not need to do resize data transform
        >>>             [dict(type='LoadOCRAnnotations', with_text=True)],
        >>>             [
        >>>                 dict(
        >>>                     type='PackTextRecogInputs',
        >>>                     meta_keys=('img_path', 'ori_shape', 'img_shape',
        >>>                                'valid_ratio'))
        >>>             ]
        >>>         ])
        >>> ]
    """

    def merge_preds(self,
                    data_samples_list: List[RecSampleList]) -> RecSampleList:
        """Merge predictions of enhanced data to one prediction.

        Args:
            data_samples_list (List[RecSampleList]): List of predictions of
                all enhanced data. The shape of data_samples_list is (B, M),
                where B is the batch size and M is the number of augmented
                data.

        Returns:
            RecSampleList: Merged prediction.
        """
        predictions = list()
        for data_samples in data_samples_list:
            scores = [
                data_sample.pred_text.score for data_sample in data_samples
            ]
            average_scores = np.array(
                [sum(score) / max(1, len(score)) for score in scores])
            max_idx = np.argmax(average_scores)
            predictions.append(data_samples[max_idx])
        return predictions

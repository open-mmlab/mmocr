# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from typing import Dict, List, Optional, Sequence

import numpy as np

from mmocr.core import TextDetDataSample
from mmocr.utils import is_type_list, rescale_polygons


class BasePostprocessor:
    """Deprecated.

    TODO: remove this class when all det postprocessors are
    refactored
    """

    def __init__(self, text_repr_type='poly'):
        assert text_repr_type in ['poly', 'quad'
                                  ], f'Invalid text repr type {text_repr_type}'

        self.text_repr_type = text_repr_type

    def is_valid_instance(self, area, confidence, area_thresh,
                          confidence_thresh):
        """If the area is a valid instance."""

        return bool(area >= area_thresh and confidence > confidence_thresh)


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
                 pred_results: dict,
                 data_samples: Sequence[TextDetDataSample],
                 training: bool = False) -> Sequence[TextDetDataSample]:
        """Postprocess pred_results according to metainfos in data_samples.

        Args:
            pred_results (dict): The prediction results stored in a dictionary.
                Usually each item to be post-processed is expected to be a
                batched tensor.
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

    def _process_single(self, pred_result: dict,
                        data_sample: TextDetDataSample,
                        **kwargs) -> TextDetDataSample:
        """Process prediction results from one image.

        Args:
            pred_result (dict): Prediction results of an image.
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

    def get_text_instances(self, pred_results: dict,
                           data_sample: TextDetDataSample,
                           **kwargs) -> TextDetDataSample:
        """Get text instance predictions of one image.

        Args:
            pred_result (dict): Prediction results of an image.
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

    def split_results(self,
                      pred_results: Dict,
                      fields: Optional[Sequence[str]] = None,
                      keep_unsplit_fields: bool = False) -> List[Dict]:
        """Split batched elements in pred_results along the first dimension
        into ``batch_num`` sub-elements and regather them into a list of dicts.

        Args:
            pred_results (dict): Raw result dictionary from detection head.
                Each item usually has the shape of (N, ...)
            fields (list[str], optional): Fields to split. If not specified,
                all fields in ``pred_results`` will be split.
            keep_unsplit_fields (bool): Whether to keep unsplit fields in
                result dicts. If True, the fields not specified in ``fields``
                will be copied to each result dict. Defaults to False.

        Returns:
            list[dict]: N dicts whose keys remains the same as that of
            pred_results.
        """
        assert isinstance(pred_results, dict) and len(pred_results) > 0
        assert fields is None or is_type_list(fields, str)
        assert isinstance(keep_unsplit_fields, bool)

        if fields is None:
            fields = list(pred_results.keys())
        batch_num = len(pred_results[fields[0]])
        results = [{} for _ in range(batch_num)]
        for field in fields:
            for i in range(batch_num):
                results[i][field] = pred_results[field][i]
        if keep_unsplit_fields:
            for k, v in pred_results.items():
                if k in fields:
                    continue
                for i in range(batch_num):
                    results[i][k] = v
        return results

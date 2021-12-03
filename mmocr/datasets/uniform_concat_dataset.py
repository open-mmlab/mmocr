# Copyright (c) OpenMMLab. All rights reserved.
import copy

from mmdet.datasets import DATASETS, ConcatDataset, build_dataset

from mmocr.utils import is_2dlist


@DATASETS.register_module()
class UniformConcatDataset(ConcatDataset):
    """A wrapper of ConcatDataset which support dataset pipeline assignment and
    replacement.

    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
        separate_eval (bool): Whether to evaluate the results
            separately if it is used as validation dataset.
            Defaults to True.
        pipeline (None | list[dict] | list[list[dict]]): If ``None``,
            each dataset in datasets use its own pipeline;
            If ``list[dict]``, it will be assigned to the dataset whose
            pipeline is None in datasets;
            If ``list[list[dict]]``, pipeline of dataset which is None
            in datasets will be replaced by the corresponding pipeline
            in the list.
    """

    def __init__(self, datasets, separate_eval=True, pipeline=None, **kwargs):
        from_cfg = all(isinstance(x, dict) for x in datasets)
        if pipeline is not None:
            assert from_cfg, 'datasets should be config dicts'
            if is_2dlist(pipeline):
                assert len(datasets) == len(pipeline)
                for dataset, tmp_pipeline in zip(datasets, pipeline):
                    assert all(isinstance(x, dict) for x in tmp_pipeline)
                    if dataset['pipeline'] is None:
                        dataset['pipeline'] = copy.deepcopy(tmp_pipeline)
            elif isinstance(pipeline, list):
                assert all(isinstance(x, dict) for x in pipeline)
                for dataset in datasets:
                    if dataset['pipeline'] is None:
                        dataset['pipeline'] = copy.deepcopy(pipeline)
        datasets = [build_dataset(c, kwargs) for c in datasets]
        super().__init__(datasets, separate_eval)

# Copyright (c) OpenMMLab. All rights reserved.
import copy

from mmdet.datasets import DATASETS, ConcatDataset, build_dataset

from mmocr.utils import is_2dlist, is_type_list


@DATASETS.register_module()
class UniformConcatDataset(ConcatDataset):
    """A wrapper of ConcatDataset which support dataset pipeline assignment and
    replacement.

    Args:
        datasets (list[dict] | list[list[dict]]): A list of datasets cfgs.
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
        force_apply (bool): If True, apply pipeline above to each dataset
            even if it have its own pipeline. Default: False.
    """

    def __init__(self,
                 datasets,
                 separate_eval=True,
                 pipeline=None,
                 force_apply=False,
                 **kwargs):
        new_datasets = []
        if pipeline is not None:
            assert len(
                pipeline
            ) > 0, 'pipeline must be list[dict] or list[list[dict]].'
            if is_type_list(pipeline, dict):
                self._apply_pipeline(datasets, pipeline, force_apply)
                new_datasets = datasets
            elif is_2dlist(pipeline):
                assert is_2dlist(datasets)
                assert len(datasets) == len(pipeline)
                for sub_datasets, tmp_pipeline in zip(datasets, pipeline):
                    self._apply_pipeline(sub_datasets, tmp_pipeline,
                                         force_apply)
                    new_datasets.extend(sub_datasets)
        else:
            if is_2dlist(datasets):
                for sub_datasets in datasets:
                    new_datasets.extend(sub_datasets)
            else:
                new_datasets = datasets
        datasets = [build_dataset(c, kwargs) for c in new_datasets]
        super().__init__(datasets, separate_eval)

    @staticmethod
    def _apply_pipeline(datasets, pipeline, force_apply=False):
        from_cfg = all(isinstance(x, dict) for x in datasets)
        assert from_cfg, 'datasets should be config dicts'
        assert all(isinstance(x, dict) for x in pipeline)
        for dataset in datasets:
            if dataset['pipeline'] is None or force_apply:
                dataset['pipeline'] = copy.deepcopy(pipeline)

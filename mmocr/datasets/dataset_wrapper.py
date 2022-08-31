# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, List, Sequence, Union

from mmengine.dataset import BaseDataset, Compose
from mmengine.dataset import ConcatDataset as MMENGINE_CONCATDATASET

from mmocr.registry import DATASETS


@DATASETS.register_module()
class ConcatDataset(MMENGINE_CONCATDATASET):
    """A wrapper of concatenated dataset.

    Same as ``torch.utils.data.dataset.ConcatDataset`` and support lazy_init.

    Note:
        ``ConcatDataset`` should not inherit from ``BaseDataset`` since
        ``get_subset`` and ``get_subset_`` could produce ambiguous meaning
        sub-dataset which conflicts with original dataset. If you want to use
        a sub-dataset of ``ConcatDataset``, you should set ``indices``
        arguments for wrapped dataset which inherit from ``BaseDataset``.

    Args:
        datasets (Sequence[BaseDataset] or Sequence[dict]): A list of datasets
            which will be concatenated.
        pipeline (list, optional): Processing pipeline to be applied to all
            of the concatenated datasets. Defaults to [].
        verify_meta (bool): Whether to verify the consistency of meta
            information of the concatenated datasets. Defaults to True.
        force_apply (bool): Whether to force apply pipeline to all datasets if
            any of them already has the pipeline configured. Defaults to False.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. Defaults to False.
    """

    def __init__(self,
                 datasets: Sequence[Union[BaseDataset, dict]],
                 pipeline: List[Union[dict, Callable]] = [],
                 verify_meta: bool = True,
                 force_apply: bool = False,
                 lazy_init: bool = False):
        self.datasets: List[BaseDataset] = []

        # Compose dataset
        pipeline = Compose(pipeline)

        for i, dataset in enumerate(datasets):
            if isinstance(dataset, dict):
                self.datasets.append(DATASETS.build(dataset))
            elif isinstance(dataset, BaseDataset):
                self.datasets.append(dataset)
            else:
                raise TypeError(
                    'elements in datasets sequence should be config or '
                    f'`BaseDataset` instance, but got {type(dataset)}')
            if len(pipeline.transforms) > 0:
                if len(self.datasets[-1].pipeline.transforms
                       ) > 0 and not force_apply:
                    raise ValueError(
                        f'The pipeline of dataset {i} is not empty, '
                        'please set `force_apply` to True.')
                self.datasets[-1].pipeline = pipeline

        self._metainfo = self.datasets[0].metainfo

        if verify_meta:
            # Only use metainfo of first dataset.
            for i, dataset in enumerate(self.datasets, 1):
                if self._metainfo != dataset.metainfo:
                    raise ValueError(
                        f'The meta information of the {i}-th dataset does not '
                        'match meta information of the first dataset')

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()
            self._metainfo.update(dict(cumulative_sizes=self.cumulative_sizes))

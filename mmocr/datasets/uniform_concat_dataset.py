import copy

from mmdet.datasets import DATASETS, ConcatDataset, build_dataset


@DATASETS.register_module()
class UniformConcatDataset(ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    concat the group flag for image aspect ratio.

    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
        separate_eval (bool): Whether to evaluate the results
            separately if it is used as validation dataset.
            Defaults to True.
    """

    def __init__(self, datasets, separate_eval=True, pipeline=None, **kwargs):
        from_cfg = all(isinstance(x, dict) for x in datasets)
        if pipeline is not None:
            assert from_cfg, 'datasets should be config dicts'
            for dataset in datasets:
                if dataset['pipeline'] is None:
                    dataset['pipeline'] = copy.deepcopy(pipeline)
        datasets = [build_dataset(c, kwargs) for c in datasets]
        super().__init__(datasets, separate_eval)

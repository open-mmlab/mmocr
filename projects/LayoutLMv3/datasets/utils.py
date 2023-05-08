from typing import Any, Mapping, Sequence

import torch
from mmengine.dataset.utils import COLLATE_FUNCTIONS
from mmengine.structures import BaseDataElement


@COLLATE_FUNCTIONS.register_module()
def long_text_data_collate(data_batch: Sequence, training: bool = True) -> Any:
    """This code is referenced from
    ``mmengine.dataset.utils.default_collate``"""
    data_item = data_batch[0]
    data_item_type = type(data_item)

    if isinstance(data_item, (BaseDataElement, str, bytes)):
        return data_batch
    elif isinstance(data_item, tuple) and hasattr(data_item, '_fields'):
        # named_tuple
        return data_item_type(*(long_text_data_collate(samples, training)
                                for samples in zip(*data_batch)))
    elif isinstance(data_item, list):
        flattened_data_batch = [
            sub_item for item in data_batch for sub_item in item
        ]
        if training:
            return flattened_data_batch[:len(data_batch)]
        else:
            return flattened_data_batch
    elif isinstance(data_item, Sequence):
        # check to make sure that the data_itements in batch have
        # consistent size
        it = iter(data_batch)
        data_item_size = len(next(it))
        if not all(len(data_item) == data_item_size for data_item in it):
            raise RuntimeError(
                'each data_itement in list of batch should be of equal size')
        transposed = list(zip(*data_batch))

        if isinstance(data_item, tuple):
            return [
                long_text_data_collate(samples, training)
                for samples in transposed
            ]  # Compat with Pytorch.
        else:
            try:
                return data_item_type([
                    long_text_data_collate(samples, training)
                    for samples in transposed
                ])
            except TypeError:
                # The sequence type may not support `__init__(iterable)`
                # (e.g., `range`).
                return [
                    long_text_data_collate(samples, training)
                    for samples in transposed
                ]
    elif isinstance(data_item, Mapping):
        return data_item_type({
            key: long_text_data_collate([d[key] for d in data_batch], training)
            for key in data_item
        })
    else:
        concat_data_batch = torch.concat(data_batch, dim=0)
        if training:
            return concat_data_batch[:len(data_batch)]
        else:
            return concat_data_batch

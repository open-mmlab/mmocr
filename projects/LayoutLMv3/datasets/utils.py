from typing import Dict, Sequence

import torch
from mmengine.dataset.utils import COLLATE_FUNCTIONS


@COLLATE_FUNCTIONS.register_module()
def ser_collate(data_batch: Sequence, training: bool = True) -> Dict:
    """A collate function designed for SER.

    Args:
        data_batch (Sequence): Data sampled from dataset.
        Like:
            [
                {
                    'inputs': {'input_ids': ..., 'bbox': ..., ...},
                    'data_samples': ['SERDataSample_1']
                },
                {
                    'inputs': {'input_ids': ..., 'bbox': ..., ...},
                    'data_samples': ['SERDataSample_1', 'SERDataSample_2', ...]
                },
                ...
            ]
        training (bool): whether training process or not.

    Note:
        Different from ``default_collate`` in pytorch or in mmengine,
        ``ser_collate`` can accept `inputs` tensor and `data_samples`
        list with the different shape.

    Returns:
        transposed (Dict): A dict have two elements,
            the first element `inputs` is a dict
            the second element `data_samples` is a list
    """
    batch_size = len(data_batch)
    # transpose `inputs`, which is a dict.
    batch_inputs = [data_item['inputs'] for data_item in data_batch]
    batch_inputs_item = batch_inputs[0]
    transposed_batch_inputs = {}
    for key in batch_inputs_item:
        concat_value = torch.concat([d[key] for d in batch_inputs], dim=0)
        # TODO: because long text will be truncated, the concat_value
        # cannot be sliced directly when training=False.
        # How to support batch inference?
        transposed_batch_inputs[key] = concat_value[:batch_size] \
            if training else concat_value
    # transpose `data_samples`, which is a list.
    batch_data_samples = [
        data_item['data_samples'] for data_item in data_batch
    ]
    flattened = [sub_item for item in batch_data_samples for sub_item in item]
    # TODO: because long text will be truncated, the concat_value
    # cannot be sliced directly when training=False.
    # How to support batch inference?
    transposed_batch_data_samples = flattened[:batch_size] \
        if training else flattened

    transposed = {
        'inputs': transposed_batch_inputs,
        'data_samples': transposed_batch_data_samples
    }
    return transposed

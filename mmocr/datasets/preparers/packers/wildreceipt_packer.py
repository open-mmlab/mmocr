# Copyright (c) OpenMMLab. All rights reserved.
import json
from typing import List

from ..data_preparer import DATA_PACKERS
from .base import BasePacker


@DATA_PACKERS.register_module()
class WildReceiptPacker(BasePacker):
    """Pack the wildreceipt annotation to MMOCR format.

    Args:
        merge_bg_others (bool): If True, give the same label to "background"
                class and "others" class. Defaults to True.
        ignore_idx (int): Index for ``ignore`` class. Defaults to 0.
        others_idx (int): Index for ``others`` class. Defaults to 25.
    """

    def __init__(self,
                 merge_bg_others: bool = False,
                 ignore_idx: int = 0,
                 others_idx: int = 25,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.ignore_idx = ignore_idx
        self.others_idx = others_idx
        self.merge_bg_others = merge_bg_others

    def add_meta(self, samples: List) -> List:
        """No meta info is required for the wildreceipt dataset."""
        return samples

    def pack_instance(self, sample: str, split: str):
        """Pack line-json str of close set to line-json str of open set.

        Args:
            sample (str): The string to be deserialized to
                the close set dictionary object.
            split (str): The split of the instance.
        """
        # Two labels at the same index of the following two lists
        # make up a key-value pair. For example, in wildreceipt,
        # closeset_key_inds[0] maps to "Store_name_key"
        # and closeset_value_inds[0] maps to "Store_addr_value".
        closeset_key_inds = list(range(2, self.others_idx, 2))
        closeset_value_inds = list(range(1, self.others_idx, 2))

        openset_node_label_mapping = {
            'bg': 0,
            'key': 1,
            'value': 2,
            'others': 3
        }
        if self.merge_bg_others:
            openset_node_label_mapping['others'] = openset_node_label_mapping[
                'bg']

        closeset_obj = json.loads(sample)
        openset_obj = {
            'file_name': closeset_obj['file_name'],
            'height': closeset_obj['height'],
            'width': closeset_obj['width'],
            'annotations': []
        }

        edge_idx = 1
        label_to_edge = {}
        for anno in closeset_obj['annotations']:
            label = anno['label']
            if label == self.ignore_idx:
                anno['label'] = openset_node_label_mapping['bg']
                anno['edge'] = edge_idx
                edge_idx += 1
            elif label == self.others_idx:
                anno['label'] = openset_node_label_mapping['others']
                anno['edge'] = edge_idx
                edge_idx += 1
            else:
                edge = label_to_edge.get(label, None)
                if edge is not None:
                    anno['edge'] = edge
                    if label in closeset_key_inds:
                        anno['label'] = openset_node_label_mapping['key']
                    elif label in closeset_value_inds:
                        anno['label'] = openset_node_label_mapping['value']
                else:
                    tmp_key = 'key'
                    if label in closeset_key_inds:
                        label_with_same_edge = closeset_value_inds[
                            closeset_key_inds.index(label)]
                    elif label in closeset_value_inds:
                        label_with_same_edge = closeset_key_inds[
                            closeset_value_inds.index(label)]
                        tmp_key = 'value'
                    edge_counterpart = label_to_edge.get(
                        label_with_same_edge, None)
                    if edge_counterpart is not None:
                        anno['edge'] = edge_counterpart
                    else:
                        anno['edge'] = edge_idx
                        edge_idx += 1
                    anno['label'] = openset_node_label_mapping[tmp_key]
                    label_to_edge[label] = anno['edge']

        openset_obj['annotations'] = closeset_obj['annotations']

        return json.dumps(openset_obj, ensure_ascii=False)

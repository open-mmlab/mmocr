# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
from typing import Any, Callable, List, Optional, Sequence, Union

from mmengine.dataset import BaseDataset

from mmocr.registry import DATASETS, TASK_UTILS

SPECIAL_TOKENS = []


@DATASETS.register_module()
class CORDDataset(BaseDataset):
    r"""CORDDataset for KIE.

    The annotation format can be jsonl. It should be a list of dicts.

    The annotation formats are shown as follows.
    - jsonl format
    .. code-block:: none

        ``{"filename": "test_img1.jpg", "ground_truth": {"OpenMMLab"}}``
        ``{"filename": "test_img2.jpg", "ground_truth": {"MMOCR"}}``

    Args:
        ann_file (str): Annotation file path. Defaults to ''.
        backend_args (dict, optional): Arguments to instantiate the
            prefix of uri corresponding backend. Defaults to None.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        data_prefix (dict): Prefix for training data. Defaults to
            ``dict(img_path='')``.
        filter_cfg (dict, optional): Config for filter data. Defaults to None.
        indices (int or Sequence[int], optional): Support using first few data
            in annotation file to facilitate training/testing on a smaller
            dataset. Defaults to None which means using all ``data_infos``.
        serialize_data (bool, optional): Whether to hold memory using
            serialized objects, when enabled, data loader workers can use
            shared RAM from master process instead of making a copy. Defaults
            to True.
        pipeline (list, optional): Processing pipeline. Defaults to [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Defaults to False.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. In some cases, such as visualization, only the meta
            information of the dataset is needed, which is not necessary to
            load annotation file. ``RecogTextDataset`` can skip load
            annotations to save time by set ``lazy_init=False``. Defaults to
            False.
        max_refetch (int, optional): If ``RecogTextDataset.prepare_data`` get a
            None img. The maximum extra number of cycles to get a valid
            image. Defaults to 1000.
    """

    def __init__(self,
                 split_name: str = '',
                 backend_args=None,
                 parser_cfg: Optional[dict] = dict(
                     type='LineJsonParser', keys=['file_name',
                                                  'ground_truth']),
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = '',
                 data_prefix: dict = dict(img_path=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000) -> None:

        self.parser = TASK_UTILS.build(parser_cfg)
        self.backend_args = backend_args
        self.split_name = split_name

        super().__init__(
            ann_file='',
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """
        data_list = []
        # dataset = load_dataset(self.data_root, split=self.split_name)
        metadata_path = osp.join(self.data_root, self.split_name,
                                 'metadata.jsonl')
        assert osp.exists(metadata_path), metadata_path
        with open(metadata_path) as f:
            metadata = f.read().strip().split('\n')
        for sample_data in metadata:
            sample = json.loads(sample_data)
            img_path = osp.join(self.data_root, self.split_name,
                                sample['file_name'])
            gt = json.loads(sample['ground_truth'])

            if 'gt_parse' in gt:
                gt_jsons = gt.pop('gt_parse')
                gt['parses_json'] = gt_jsons
            else:
                gt['parses_json'] = gt.pop('gt_parses')

            if self.split_name == 'train':
                global SPECIAL_TOKENS
                SPECIAL_TOKENS += self.search_special_tokens(gt['parses_json'])

            if isinstance(gt, list):
                instances = gt
            else:
                instances = [gt]
            data_list.append({'img_path': img_path, 'instances': instances})
        return data_list

    def search_special_tokens(self, obj: Any, sort_json_key: bool = True):
        """Convert an ordered JSON object into a token sequence."""
        special_tokens = []
        if type(obj) == dict:
            if len(obj) == 1 and 'text_sequence' in obj:
                pass
            else:
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    special_tokens += [fr'<s_{k}>', fr'</s_{k}>']
                    special_tokens += self.search_special_tokens(obj[k])
        elif type(obj) == list:
            for item in obj:
                special_tokens += self.search_special_tokens(
                    item, sort_json_key)
        return special_tokens

# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Callable, List, Optional, Sequence, Union

from mmengine.dataset import BaseDataset
from mmengine.fileio import list_from_file
from mmocr.registry import DATASETS, TASK_UTILS

# TODO: replace all list_from_file from mmengine


@DATASETS.register_module()
class RecogTextDataset(BaseDataset):
    r"""RecogTextDataset for text recognition.

    The annotation format can be both in jsonl and txt. If the annotation file
    is in jsonl format, it should be a list of dicts. If the annotation file
    is in txt format, it should be a list of lines.

    The annotation formats are shown as follows.
    - txt format
    .. code-block:: none

        ``test_img1.jpg OpenMMLab``
        ``test_img2.jpg MMOCR``

    - jsonl format
    .. code-block:: none

        ``{"filename": "test_img1.jpg", "text": "OpenMMLab"}``
        ``{"filename": "test_img2.jpg", "text": "MMOCR"}``

    Args:
        ann_file (str): Annotation file path. Defaults to ''.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Default: None.
        parse_cfg (dict, optional): Config of parser for parsing annotations.
            Use ``LineJsonParser`` when the annotation file is in jsonl format
            with keys of ``filename`` and ``text``. The keys in parse_cfg
            should be consistent with the keys in jsonl annotations. The first
            key in parse_cfg should be the key of the path in jsonl
            annotations. The second key in parse_cfg should be the key of the
            text in jsonl Use ``LineStrParser`` when the annotation file is in
            txt format. Defaults to
            ``dict(type='LineJsonParser', keys=['filename', 'text'])``.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        data_prefix (dict): Prefix for training data. Defaults to
            ``dict(img_path='')``.
        filter_cfg (dict, optional): Config for filter data. Defaults to None.
        indices (int or Sequence[int], optional): Support using first few
            data in annotation file to facilitate training/testing on a smaller
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
                 ann_file: str = '',
                 file_client_args=None,
                 parser_cfg: Optional[dict] = dict(
                     type='LineJsonParser', keys=['filename', 'text']),
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
        self.file_client_args = file_client_args
        super().__init__(
            ann_file=ann_file,
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
        raw_anno_infos = list_from_file(
            self.ann_file, file_client_args=self.file_client_args)
        for raw_anno_info in raw_anno_infos:
            data_list.append(self.parse_data_info(raw_anno_info))
        return data_list

    def parse_data_info(self, raw_anno_info: str) -> dict:
        """Parse raw annotation to target format.

        Args:
            raw_anno_info (str): One raw data information loaded
                from ``ann_file``.

        Returns:
            (dict): Parsed annotation.
        """
        data_info = {}
        parsed_anno = self.parser(raw_anno_info)
        img_path = osp.join(self.data_prefix['img_path'],
                            parsed_anno[self.parser.keys[0]])

        data_info['img_path'] = img_path
        data_info['instances'] = [dict(text=parsed_anno[self.parser.keys[1]])]
        return data_info

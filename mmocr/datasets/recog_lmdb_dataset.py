# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import mmcv

from mmengine.dataset import BaseDataset
from mmocr.registry import DATASETS


@DATASETS.register_module()
class RecogLMDBDataset(BaseDataset):
    r"""RecogLMDBDataset for text recognition.

    The annotation format should be in lmdb format. The lmdb file should
    contain three keys: 'num-samples', 'label-xxxxxxxxx' and 'image-xxxxxxxxx',
    where 'xxxxxxxxx' is the index of the image. The value of 'num-samples' is
    the total number of images. The value of 'label-xxxxxxx' is the text label
    of the image, and the value of 'image-xxxxxxx' is the image data.

    following keys:
    Each item fetched from this dataset will be a dict containing the
    following keys:

        - img (ndarray): The loaded image.
        - img_path (str): The image key.
        - instances (list[dict]): The list of annotations for the image.

    Args:
        ann_file (str): Annotation file path. Defaults to ''.
        img_color_type (str): The flag argument for :func:``mmcv.imfrombytes``,
            which determines how the image bytes will be parsed. Defaults to
            'color'.
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
            load annotation file. ``RecogLMDBDataset`` can skip load
            annotations to save time by set ``lazy_init=False``.
            Defaults to False.
        max_refetch (int, optional): If ``RecogLMDBdataset.prepare_data`` get a
            None img. The maximum extra number of cycles to get a valid
            image. Defaults to 1000.
    """

    def __init__(
        self,
        ann_file: str = '',
        img_color_type: str = 'color',
        metainfo: Optional[dict] = None,
        data_root: Optional[str] = '',
        data_prefix: dict = dict(img_path=''),
        filter_cfg: Optional[dict] = None,
        indices: Optional[Union[int, Sequence[int]]] = None,
        serialize_data: bool = True,
        pipeline: List[Union[dict, Callable]] = [],
        test_mode: bool = False,
        lazy_init: bool = False,
        max_refetch: int = 1000,
    ) -> None:

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

        self.color_type = img_color_type

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """
        if not hasattr(self, 'env'):
            self._make_env()
            with self.env.begin(write=False) as txn:
                self.total_number = int(
                    txn.get(b'num-samples').decode('utf-8'))

        data_list = []
        with self.env.begin(write=False) as txn:
            for i in range(self.total_number):
                idx = i + 1
                label_key = f'label-{idx:09d}'
                img_key = f'image-{idx:09d}'
                text = txn.get(label_key.encode('utf-8')).decode('utf-8')
                line = [img_key, text]
                data_list.append(self.parse_data_info(line))
        return data_list

    def parse_data_info(self,
                        raw_anno_info: Tuple[Optional[str],
                                             str]) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_anno_info (str): One raw data information loaded
                from ``ann_file``.

        Returns:
            (dict): Parsed annotation.
        """
        data_info = {}
        img_key, text = raw_anno_info
        data_info['img_path'] = img_key
        data_info['instances'] = [dict(text=text)]
        return data_info

    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_info = self.get_data_info(idx)
        with self.env.begin(write=False) as txn:
            img_bytes = txn.get(data_info['img_path'].encode('utf-8'))
            if img_bytes is None:
                return None
            data_info['img'] = mmcv.imfrombytes(
                img_bytes, flag=self.color_type)
        return self.pipeline(data_info)

    def _make_env(self):
        """Create lmdb environment from self.ann_file and save it to
        ``self.env``.

        Returns:
            Lmdb environment.
        """
        try:
            import lmdb
        except ImportError:
            raise ImportError(
                'Please install lmdb to enable RecogLMDBDataset.')
        if hasattr(self, 'env'):
            return

        self.env = lmdb.open(
            self.ann_file,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def close(self):
        """Close lmdb environment."""
        if hasattr(self, 'env'):
            self.env.close()
            del self.env

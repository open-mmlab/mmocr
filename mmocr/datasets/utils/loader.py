# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmocr.datasets.builder import LOADERS, build_parser
from .backend import (HardDiskAnnFileBackend, HTTPAnnFileBackend,
                      PetrelAnnFileBackend)


class AnnFileLoader:
    """Annotation file loader to load annotations from ann_file, and parse raw
    annotation to dict format with certain parser.

    Args:
        ann_file (str): Annotation file path.
        parser (dict): Dictionary to construct parser
            to parse original annotation infos.
        repeat (int|float): Repeated times of dataset.
        file_storage_backend (str): The storage backend type for annotation
            file. Options are "disk", "http" and "petrel". Default: "disk".
        file_format (str): The format of annotation file. Options are
            "txt" and "lmdb". Default: "txt".
    """

    _backends = {
        'disk': HardDiskAnnFileBackend,
        'petrel': PetrelAnnFileBackend,
        'http': HTTPAnnFileBackend
    }

    def __init__(self,
                 ann_file,
                 parser,
                 repeat=1,
                 file_storage_backend='disk',
                 file_format='txt',
                 **kwargs):
        assert isinstance(ann_file, str)
        assert isinstance(repeat, (int, float))
        assert isinstance(parser, dict)
        assert repeat > 0
        assert file_storage_backend in ['disk', 'http', 'petrel']
        assert file_format in ['txt', 'lmdb']

        self.parser = build_parser(parser)
        self.repeat = repeat
        self.ann_file_backend = self._backends[file_storage_backend](
            file_format, **kwargs)
        self.ori_data_infos = self._load(ann_file)

    def __len__(self):
        return int(len(self.ori_data_infos) * self.repeat)

    def _load(self, ann_file):
        """Load annotation file."""

        return self.ann_file_backend(ann_file)

    def __getitem__(self, index):
        """Retrieve anno info of one instance with dict format."""
        return self.parser.get_item(self.ori_data_infos, index)

    def __iter__(self):
        self._n = 0
        return self

    def __next__(self):
        if self._n < len(self):
            data = self[self._n]
            self._n += 1
            return data
        raise StopIteration

    def close(self):
        """For ann_file with lmdb format only."""
        self.ori_data_infos.close()


@LOADERS.register_module()
class HardDiskLoader(AnnFileLoader):
    """Load txt format annotation file from hard disks."""

    def __init__(self, ann_file, parser, repeat=1):
        warnings.warn(
            'HardDiskLoader is deprecated, please use '
            'AnnFileLoader instead.', UserWarning)
        super().__init__(
            ann_file,
            parser,
            repeat,
            file_storage_backend='disk',
            file_format='txt')


@LOADERS.register_module()
class LmdbLoader(AnnFileLoader):
    """Load lmdb format annotation file from hard disks."""

    def __init__(self, ann_file, parser, repeat=1):
        warnings.warn(
            'LmdbLoader is deprecated, please use '
            'AnnFileLoader instead.', UserWarning)
        super().__init__(
            ann_file,
            parser,
            repeat,
            file_storage_backend='disk',
            file_format='lmdb')

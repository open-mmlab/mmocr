# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import os.path as osp
import shutil
import warnings

import mmcv

from mmocr import digit_version
from mmocr.utils import list_from_file


class LmdbAnnFileBackend:
    """Lmdb storage backend for annotation file.

    Args:
        lmdb_path (str): Lmdb file path.
    """

    def __init__(self, lmdb_path, file_format, encoding='utf8'):
        self.lmdb_path = lmdb_path
        self.encoding = encoding
        self.file_format = file_format
        env = self._get_env()
        with env.begin(write=False) as txn:
            self.total_number = int(
                txn.get('num-samples'.encode('utf-8')).decode(self.encoding))
            try:
                image_key = f'image-{1:09d}'
                txn.get(image_key.encode(encoding))
                self.label_only = False
            except Exception:
                self.label_only = True

    def __getitem__(self, index):
        """Retrieve one line from lmdb file by index."""
        if not hasattr(self, 'env'):
            self.env = self._get_env()

        with self.env.begin(write=False) as txn:
            index = index + 1
            label_key = f'label-{index:09d}'
            if self.label_only:
                line = txn.get(label_key.encode('utf-8')).decode(self.encoding)

            else:
                img_key = f'image-{index:09d}'
                text = txn.get(label_key.encode('utf-8')).decode(self.encoding)
                if self.file_format == 'txt':
                    line = img_key + ' ' + text
                else:
                    line = json.dumps({'filename': img_key, 'text': text})
            return line

    def __len__(self):
        return self.total_number

    def _get_env(self):
        try:
            import lmdb
        except ImportError:
            raise ImportError(
                'Please install lmdb to enable LmdbAnnFileBackend.')
        return lmdb.open(
            self.lmdb_path,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def close(self):
        self.env.close()


class HardDiskAnnFileBackend:
    """Load annotation file with raw hard disks storage backend."""

    def __init__(self, file_format='txt'):
        assert file_format in ['txt', 'lmdb']
        self.file_format = file_format

    def __call__(self, ann_file):
        if self.file_format == 'lmdb':
            return LmdbAnnFileBackend(ann_file, self.file_format)

        return list_from_file(ann_file)


class PetrelAnnFileBackend:
    """Load annotation file with petrel storage backend."""

    def __init__(self, file_format='txt', save_dir='tmp_dir'):
        assert file_format in ['txt', 'lmdb']
        self.file_format = file_format
        self.save_dir = save_dir

    def __call__(self, ann_file):
        file_client = mmcv.FileClient(backend='petrel')

        if self.file_format == 'lmdb':
            mmcv_version = digit_version(mmcv.__version__)
            if mmcv_version < digit_version('1.3.16'):
                raise Exception('Please update mmcv to 1.3.16 or higher '
                                'to enable "get_local_path" of "FileClient".')
            assert file_client.isdir(ann_file)
            files = file_client.list_dir_or_file(ann_file)

            ann_file_rel_path = ann_file.split('s3://')[-1]
            ann_file_dir = osp.dirname(ann_file_rel_path)
            ann_file_name = osp.basename(ann_file_rel_path)
            local_dir = osp.join(self.save_dir, ann_file_dir, ann_file_name)
            if osp.exists(local_dir):
                warnings.warn(
                    f'local_ann_file: {local_dir} is already existed and '
                    'will be used. If it is not the correct ann_file '
                    'corresponding to {ann_file}, please remove it or '
                    'change "save_dir" first then try again.')
            else:
                os.makedirs(local_dir, exist_ok=True)
                print(f'Fetching {ann_file} to {local_dir}...')
                for each_file in files:
                    tmp_file_path = file_client.join_path(ann_file, each_file)
                    with file_client.get_local_path(
                            tmp_file_path) as local_path:
                        shutil.copy(local_path, osp.join(local_dir, each_file))

            return LmdbAnnFileBackend(local_dir, self.file_format)

        lines = str(file_client.get(ann_file), encoding='utf-8').split('\n')

        return [x for x in lines if x.strip() != '']


class HTTPAnnFileBackend:
    """Load annotation file with http storage backend."""

    def __init__(self, file_format='txt'):
        assert file_format in ['txt', 'lmdb']
        self.file_format = file_format

    def __call__(self, ann_file):
        file_client = mmcv.FileClient(backend='http')

        if self.file_format == 'lmdb':
            raise NotImplementedError(
                'Loading lmdb file on http is not supported yet.')

        lines = str(file_client.get(ann_file), encoding='utf-8').split('\n')

        return [x for x in lines if x.strip() != '']

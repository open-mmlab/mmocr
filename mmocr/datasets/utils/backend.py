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

    def __init__(self, lmdb_path, encoding='utf8'):
        """Currently we support two lmdb formats, one is the lmdb file with
        only abels generated by txt2lmdb (deprecated), and one is the lmdb file
        generated by recog2lmdb.

        The former stores string in 'filename text' format directly in lmdb,
        while the latter uses a more reasonable image_key as well as label_key
        for querying.
        """
        self.lmdb_path = lmdb_path
        self.encoding = encoding
        self.deprecated_format = False
        env = self._get_env()
        with env.begin(write=False) as txn:
            try:
                self.total_number = int(
                    txn.get('num-samples'.encode('utf-8')).decode(
                        self.encoding))
            except AttributeError:
                warnings.warn(
                    'DeprecationWarning: The lmdb dataset generated with '
                    'txt2lmdb will be deprecate, please use the latest '
                    'tools/data/utils/recog2lmdb to generate the lmdb dataset')
                self.total_number = int(
                    txn.get('total_number'.encode('utf-8')).decode(
                        self.encoding))
                self.deprecated_format = True
            # The lmdb file may contain only the label, or it may contain both
            # the label and the image, so we use image_key here for probing.
            try:
                image_key = f'image-{1:09d}'
                txn.get(image_key.encode(encoding))
                self.label_only = False
            except Exception:
                self.label_only = True

    def __getitem__(self, index):
        """Retrieve one line from lmdb file by index.

        In order to support space
        reading, the returned lines are in the form of json, such as
        '{'filename': 'image1.jpg' ,'text':'HELLO'}'
        """
        if not hasattr(self, 'env'):
            self.env = self._get_env()

        with self.env.begin(write=False) as txn:
            if self.deprecated_format:
                line = txn.get(str(index).encode('utf-8')).decode(
                    self.encoding)
                filename, text = line.split(' ')
                line = json.dumps({
                    'filename': filename,
                    'text': text
                },
                                  ensure_ascii=False)
            else:
                index = index + 1
                label_key = f'label-{index:09d}'
                if self.label_only:
                    line = txn.get(label_key.encode('utf-8')).decode(
                        self.encoding)
                    line = json.dumps(line, ensure_ascii=False)
                else:
                    img_key = f'image-{index:09d}'
                    text = txn.get(label_key.encode('utf-8')).decode(
                        self.encoding)
                    line = json.dumps({
                        'filename': img_key,
                        'text': text
                    },
                                      ensure_ascii=False)
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
            return LmdbAnnFileBackend(ann_file)

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

            return LmdbAnnFileBackend(local_dir)

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

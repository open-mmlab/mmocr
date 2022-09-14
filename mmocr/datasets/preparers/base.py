# Copyright (c) OpenMMLab. All rights reserved.
import hashlib
import json
import math
import os.path as osp
import shutil
import ssl
import sys
import time
import urllib.request as request
from glob import glob
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from mmengine.config import Config
from mmengine.utils import mkdir_or_exist

from mmocr.utils import bbox2poly, poly2bbox

ssl._create_default_https_context = ssl._create_unverified_context


class BaseDatasetPreparer:
    """Base class of dataset preparer.

    Dataset preparer is used to prepare dataset for MMOCR. It mainly consists
    of three steps:

        1. Download and extract dataset files.
        2. Convert annotations to MMOCR format.
        3. Dump the converted labels (images) to the disk.

    Check out the dataset format used in MMOCR here:
    https://mmocr.readthedocs.io/en/dev-1.x/user_guides/dataset_prepare.html
    """

    def __init__(self,
                 dataset_name: str,
                 data_root: str = 'data/',
                 task: str = 'det',
                 nproc: int = 4) -> None:
        """Initialization.

        Args:
            dataset_name (str): Dataset name.
            data_root (str): Path to data root.
            task (str): Task type. Options are 'det', 'rec', and 'spot'.
            nproc (int): Number of parallel processes.
        """
        self.cfg_file = f'dataset_zoo/{dataset_name}.yml'
        if not osp.exists(self.cfg_file):
            raise FileNotFoundError(f'Config file {self.cfg_file} not found.')
        self.data_root = osp.join(data_root, task, dataset_name)
        self.cfg = Config.fromfile(self.cfg_file)
        self.task = task
        self.splits = self.cfg.Data.Splits
        self.ignore_flag = self.cfg.Data.Annotation.Ignore
        self.nproc = nproc
        self.data_list = dict()
        assert self.task in self.cfg.Task, \
            f'{task} not supported for {self.cfg.Name}'
        if self.cfg.Data.License.Type:
            print(f'\033[1;33;40mDataset Name: {self.cfg.Name}')
            print(f'License Type: {self.cfg.Data.License.Type}')
            print(f'License Link: {self.cfg.Data.License.Link}')
            print(f'BibTeX: {self.cfg.Paper.BibTeX}\033[0m')
            print(
                '\033[1;31;43mMMOCR does not own the dataset. Using this '
                'dataset you must accept the license provided by the owners, '
                'and cite the corresponding papers appropriately.')
            print('If you do not agree with the above license, please cancel '
                  'the progress immediately by pressing ctrl+c. Otherwise, '
                  'you are deemed to accept the terms and conditions.\033[0m')
            time.sleep(5)
        mkdir_or_exist(osp.join(self.data_root, 'imgs'))
        mkdir_or_exist(osp.join(self.data_root, 'annotations'))

    def process(self):
        """Prepare the dataset."""
        self._download()
        self._move()
        self._convert()
        self._dump()

    def _download(self) -> None:
        """Download and extract dataset files."""

        def iszip(file_path: str) -> bool:
            """Check whether the file is a zip.

            Args:
                file_path (str): Path to the file.

            Returns:
                bool: Whether the file is a zip.
            """

            suffixes = ['zip', 'tar', 'tar.gz']
            for suffix in suffixes:
                if file_path.endswith(suffix):
                    return True
            return False

        def extract(path: str) -> None:
            """Extract downloaded dataset zip/tar.gz files.

            Args:
                path (str): Path to the zip file.
            """

            dst_path = osp.splitext(path)[0]
            mkdir_or_exist(dst_path)
            print(f'Try to extract file: {osp.basename(path)}')
            if path.endswith('.zip'):
                try:
                    import zipfile
                except ImportError:
                    raise ImportError('Please install zipfile.')
                with zipfile.ZipFile(path, 'r') as zip_ref:
                    zip_ref.extractall(dst_path)
            elif path.endswith('.tar.gz') or path.endswith('.tar'):
                try:
                    import tarfile
                except ImportError:
                    raise ImportError('Please install tarfile.')
                with tarfile.open(path, 'r:gz') as tar_ref:
                    tar_ref.extractall(dst_path)
            else:
                raise NotImplementedError(f'Unsupported file format {path}')

        def progress(down: float, block: float, size: float) -> None:
            """Show download progress.

            Args:
                down (float): Downloaded size.
                block (float): Block size.
                size (float): Total size of the file.
            """

            percent = min(100. * down * block / size, 100)
            print(f'\rDownload progress: {percent:.2f}%', end='')

        links = self.cfg.Link[self.task.capitalize()]
        image_links = links.Image.Train + links.Image.Test + links.Image.Val
        anno_links = links.Annotation.Train + links.Annotation.Test + \
            links.Annotation.Val
        for link in image_links + anno_links:
            if link:
                link, md5 = link
                file_name = osp.basename(link)
                dst_path = osp.join(self.data_root, file_name)
                if not (osp.exists(dst_path)
                        and self._check_md5(dst_path, md5)):
                    print(f'Downloading {file_name}')
                    request.urlretrieve(link, dst_path, progress)
                    if iszip(dst_path):
                        extract(dst_path)

    def _move(self) -> None:
        """Rename and move dataset files."""

        for src, dst in self.file_mapper:
            src = osp.join(self.data_root, src)
            dst = osp.join(self.data_root, dst)
            if osp.exists(src) and not osp.exists(dst):
                shutil.move(src, dst)

    def _convert(self) -> None:
        """Convert annotations to MMOCR format."""

        raise NotImplementedError

    def _dump(self) -> None:
        """Dump annotations to json file."""

        for split in self.splits:
            if self.task == 'det':
                dataset, dst_file = self._pack_det_dataset(split)
            if self.task == 'rec':
                dataset, dst_file = self._pack_rec_dataset(split)

            with open(osp.join(self.data_root, dst_file), 'w') as f:
                json.dump(dataset, f)
            print(f'Successfully dumping the annotation file to {dst_file}')

    def _retrieve_img_list(self, split: str) -> List:
        """Retrieve image list from split.

        Args:
            split (str): Split of the dataset.
        """

        suffixes = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG']
        img_list = []
        img_dir = osp.join(self.data_root, 'imgs', split)
        for suffix in suffixes:
            img_list.extend(glob(osp.join(img_dir, '*' + suffix)))

        return img_list

    @staticmethod
    def _pack_det_instance(box: Optional[Sequence] = None,
                           poly: Optional[Sequence] = None,
                           bbox_label: int = 0,
                           ignore: bool = False) -> Dict:
        """Packing the info to a detection instance.

        Args:
            box (Optional[Sequence], optional): Bounding box. Defaults to None.
            poly (Optional[Sequence], optional): Polygon. Defaults to None.
            bbox_label (int, optional): Bounding box label. Defaults to 0.
            ignore (bool, optional): Ignore flag. Defaults to False.

        Returns:
            Dict: The packed instance.
        """

        assert box or poly, "box and poly can't be both None"
        return {
            'polygon':
            poly if poly else list(bbox2poly(box).astype('float64')),
            'bbox': box if box else list(poly2bbox(poly).astype('float64')),
            'bbox_label': bbox_label,
            'ignore': ignore
        }

    @staticmethod
    def _pack_rec_instance(text: str) -> Dict:
        """Packing the info to a recognition instance.

        Args:
            text (str): Text content.

        Returns:
            Dict: The packed instance.
        """

        return {'instances': [{'text': text}]}

    def _pack_det_dataset(self, split: str) -> Tuple:
        """Pack the annotations to MMOCR dataset format.

        Args:
            split (str): Split of the dataset.

        Returns:
            Tuple: A tuple containing the packed dataset and the file name.
        """

        mapping = {'train': 'training', 'test': 'test', 'val': 'val'}
        dataset = {
            'metainfo': {
                'dataset_type': 'TextDetDataset',
                'task_name': 'textdet',
                'category': [{
                    'id': 0,
                    'name': 'text'
                }]
            },
            'data_list': self.data_list[split]
        }
        return dataset, f'instances_{mapping[split]}.json'

    def _pack_rec_dataset(self, split: str) -> Tuple:
        """Pack the annotations to MMOCR dataset format.

        Args:
            split (str): Split of the dataset.

        Returns:
            Tuple: A tuple containing the packed dataset and the file name.
        """

        dataset = {
            'metainfo': {
                'dataset_type': 'TextRecogDataset',
                'task_name': 'textrecog'
            },
            'data_list': self.data_list[split]
        }

        return dataset, f'{split}_labels.json'

    @staticmethod
    def _crop_patch(img: np.ndarray, bbox: Sequence) -> np.ndarray:
        """Cropping images to patches for text recognition task."""

        x, y, w, h = bbox
        x, y = max(0, math.floor(x)), max(0, math.floor(y))
        w, h = math.ceil(w), math.ceil(h)

        return img[y:y + h, x:x + w]

    @staticmethod
    def _check_md5(
        file_path: str,
        md5: str,
        chunk_size: int = 1024 * 1024,
    ) -> bool:
        """Check MD5 of the file.

        Args:
            file_path (str): Path to the file.
            md5 (str): MD5 to be matched.
            chunk_size (int, optional): Chunk size. Defaults to 1024*1024.

        Returns:
            bool: Whether the md5 is matched.
        """

        if sys.version_info >= (3, 9):
            hash = hashlib.md5(usedforsecurity=False)
        else:
            hash = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                hash.update(chunk)

        return hash.hexdigest() == md5

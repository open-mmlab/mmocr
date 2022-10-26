# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import shutil
import ssl
import urllib.request as request
from typing import Dict, List, Optional, Tuple

from mmengine import mkdir_or_exist

from mmocr.utils import check_integrity, is_archive
from .data_preparer import DATA_OBTAINERS

ssl._create_default_https_context = ssl._create_unverified_context


@DATA_OBTAINERS.register_module()
class NaiveDataObtainer:
    """A naive pipeline for obtaining dataset.

    download -> extract -> move

    Args:
        files (list[dict]): A list of file information.
        cache_path (str): The path to cache the downloaded files.
        data_root (str): The root path of the dataset.
    """

    def __init__(self, files: List[Dict], cache_path: str,
                 data_root: str) -> None:
        self.files = files
        self.cache_path = cache_path
        self.data_root = data_root
        mkdir_or_exist(osp.join(self.data_root, 'imgs'))
        mkdir_or_exist(osp.join(self.data_root, 'annotations'))
        mkdir_or_exist(self.cache_path)

    def __call__(self):
        for file in self.files:
            save_name, url, md5 = file['save_name'], file['url'], file['md5']
            download_path = osp.join(
                self.cache_path,
                osp.basename(url) if save_name is None else save_name)
            # Download required files
            if not check_integrity(download_path, md5):
                self.download(url=url, dst_path=download_path)
            # Extract downloaded zip files to data root
            self.extract(src_path=download_path, dst_path=self.data_root)
            # Move & Rename dataset files
            if 'mapping' in file:
                self.move(mapping=file['mapping'])
        self.clean()

    def download(self, url: Optional[str], dst_path: str) -> None:
        """Download file from given url with progress bar.

        Args:
            url (str): The url to download the file.
            dst_path (str): The destination path to save the file.
        """

        def progress(down: float, block: float, size: float) -> None:
            """Show download progress.

            Args:
                down (float): Downloaded size.
                block (float): Block size.
                size (float): Total size of the file.
            """

            percent = min(100. * down * block / size, 100)
            file_name = osp.basename(dst_path)
            print(f'\rDownloading {file_name}: {percent:.2f}%', end='')

        if not url and not osp.exists(dst_path):
            raise FileNotFoundError(
                'Direct url is not available for this dataset.'
                ' Please manually download the required files'
                ' following the guides.')

        request.urlretrieve(url, dst_path, progress)

    def extract(self,
                src_path: str,
                dst_path: str,
                delete: bool = False) -> None:
        """Extract zip/tar.gz files.

        Args:
            src_path (str): Path to the zip file.
            dst_path (str): Path to the destination folder.
            delete (bool, optional): Whether to delete the zip file. Defaults
                to False.
        """

        if not is_archive(src_path):
            # Move the file to the destination folder if it is not a zip
            shutil.move(src_path, dst_path)
            return

        zip_name = osp.basename(src_path).split('.')[0]
        if dst_path is None:
            dst_path = osp.join(osp.dirname(src_path), zip_name)
        else:
            dst_path = osp.join(dst_path, zip_name)
        mkdir_or_exist(dst_path)
        print(f'Extracting: {osp.basename(src_path)}')
        if src_path.endswith('.zip'):
            try:
                import zipfile
            except ImportError:
                raise ImportError(
                    'Please install zipfile by running "pip install zipfile".')
            with zipfile.ZipFile(src_path, 'r') as zip_ref:
                zip_ref.extractall(dst_path)
        elif src_path.endswith('.tar.gz') or src_path.endswith('.tar'):
            if src_path.endswith('.tar.gz'):
                mode = 'r:gz'
            elif src_path.endswith('.tar'):
                mode = 'r:'
            try:
                import tarfile
            except ImportError:
                raise ImportError(
                    'Please install tarfile by running "pip install tarfile".')
            with tarfile.open(src_path, mode) as tar_ref:
                tar_ref.extractall(dst_path)
        if delete:
            os.remove(src_path)

    def move(self, mapping: List[Tuple[str, str]]) -> None:
        """Rename and move dataset files one by one.

        Args:
            mapping (List[Tuple[str, str]]): A list of tuples, each
            tuple contains the source file name and the destination file name.
        """
        for src, dst in mapping:
            src = osp.join(self.data_root, src)
            dst = osp.join(self.data_root, dst)
            if osp.exists(src) and not osp.exists(dst):
                shutil.move(src, dst)

    def clean(self) -> None:
        """Remove empty dirs."""
        for root, dirs, files in os.walk(self.data_root, topdown=False):
            if not files and not dirs:
                os.rmdir(root)

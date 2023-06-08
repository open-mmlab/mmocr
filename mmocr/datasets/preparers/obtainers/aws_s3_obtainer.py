# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import ssl
from typing import Dict, List, Optional

from mmengine import mkdir_or_exist

from mmocr.registry import DATA_OBTAINERS
from .naive_data_obtainer import NaiveDataObtainer

ssl._create_default_https_context = ssl._create_unverified_context


@DATA_OBTAINERS.register_module()
class AWSS3Obtainer(NaiveDataObtainer):
    """A AWS S3 obtainer.

    download -> extract -> move

    Args:
        files (list[dict]): A list of file information.
        cache_path (str): The path to cache the downloaded files.
        data_root (str): The root path of the dataset. It is usually set auto-
            matically and users do not need to set it manually in config file
            in most cases.
        task (str): The task of the dataset. It is usually set automatically
            and users do not need to set it manually in config file
            in most cases.
    """

    def __init__(self, files: List[Dict], cache_path: str, data_root: str,
                 task: str) -> None:
        try:
            import boto3
            from botocore import UNSIGNED
            from botocore.config import Config
        except ImportError:
            raise ImportError(
                'Please install boto3 to download hiertext dataset.')
        self.files = files
        self.cache_path = cache_path
        self.data_root = data_root
        self.task = task
        self.s3_client = boto3.client(
            's3', config=Config(signature_version=UNSIGNED))
        self.total_length = 0
        mkdir_or_exist(self.data_root)
        mkdir_or_exist(osp.join(self.data_root, f'{task}_imgs'))
        mkdir_or_exist(osp.join(self.data_root, 'annotations'))
        mkdir_or_exist(self.cache_path)

    def find_bucket_key(self, s3_path: str):
        """This is a helper function that given an s3 path such that the path
        is of the form: bucket/key It will return the bucket and the key
        represented by the s3 path.

        Args:
            s3_path (str): The AWS s3 path.
        """
        s3_components = s3_path.split('/', 1)
        bucket = s3_components[0]
        s3_key = ''
        if len(s3_components) > 1:
            s3_key = s3_components[1]
        return bucket, s3_key

    def s3_download(self, s3_bucket: str, s3_object_key: str, dst_path: str):
        """Download file from given s3 url with progress bar.

        Args:
            s3_bucket (str): The s3 bucket to download the file.
            s3_object_key (str): The s3 object key to download the file.
            dst_path (str): The destination path to save the file.
        """
        meta_data = self.s3_client.head_object(
            Bucket=s3_bucket, Key=s3_object_key)
        total_length = int(meta_data.get('ContentLength', 0))
        downloaded = 0

        def progress(chunk):
            nonlocal downloaded
            downloaded += chunk
            percent = min(100. * downloaded / total_length, 100)
            file_name = osp.basename(dst_path)
            print(f'\rDownloading {file_name}: {percent:.2f}%', end='')

        print(f'Downloading {dst_path}')
        self.s3_client.download_file(
            s3_bucket, s3_object_key, dst_path, Callback=progress)

    def download(self, url: Optional[str], dst_path: str) -> None:
        """Download file from given url with progress bar.

        Args:
            url (str): The url to download the file.
            dst_path (str): The destination path to save the file.
        """
        if url is None and not osp.exists(dst_path):
            raise FileNotFoundError(
                'Direct url is not available for this dataset.'
                ' Please manually download the required files'
                ' following the guides.')

        if url.startswith('magnet'):
            raise NotImplementedError('Please use any BitTorrent client to '
                                      'download the following magnet link to '
                                      f'{osp.abspath(dst_path)} and '
                                      f'try again.\nLink: {url}')

        print('Downloading...')
        print(f'URL: {url}')
        print(f'Destination: {osp.abspath(dst_path)}')
        print('If you stuck here for a long time, please check your network, '
              'or manually download the file to the destination path and '
              'run the script again.')
        if url.startswith('s3://'):
            url = url[5:]
            bucket, key = self.find_bucket_key(url)
            self.s3_download(bucket, key, osp.abspath(dst_path))
        elif url.startswith('https://') or url.startswith('http://'):
            super().download(url, dst_path)
        print('')

# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Optional

from mmocr.registry import DATA_OBTAINERS
from .naive_data_obtainer import NaiveDataObtainer


@DATA_OBTAINERS.register_module()
class MagnetObtainer(NaiveDataObtainer):
    """A obtainer that obtain files via magnet link.

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

    def download(self, url: Optional[str], dst_path: str) -> None:
        """Download file from given url with progress bar.

        Args:
            url (str): The url to download the file.
            dst_path (str): The destination path to save the file.
        """
        raise NotImplementedError('Please use any torrent client to download '
                                  'the file from the following link to '
                                  f'{osp.abspath(dst_path)} and '
                                  'try again.\n'
                                  f'Magnet link: {url}')

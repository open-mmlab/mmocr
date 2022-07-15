# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.utils import print_log
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose
from torch.utils.data import Dataset

from mmocr.datasets.builder import build_loader


@DATASETS.register_module()
class BaseDataset(Dataset):
    """Custom dataset for text detection, text recognition, and their
    downstream tasks.

    1. The text detection annotation format is as follows:
       The `annotations` field is optional for testing
       (this is one line of anno_file, with line-json-str
       converted to dict for visualizing only).

        .. code-block:: json

            {
                "file_name": "sample.jpg",
                "height": 1080,
                "width": 960,
                "annotations":
                    [
                        {
                            "iscrowd": 0,
                            "category_id": 1,
                            "bbox": [357.0, 667.0, 804.0, 100.0],
                            "segmentation": [[361, 667, 710, 670,
                                              72, 767, 357, 763]]
                        }
                    ]
            }

    2. The two text recognition annotation formats are as follows:
       The `x1,y1,x2,y2,x3,y3,x4,y4` field is used for online crop
       augmentation during training.

        format1: sample.jpg hello
        format2: sample.jpg 20 20 100 20 100 40 20 40 hello

    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        loader (dict): Dictionary to construct loader
            to load annotation infos.
        img_prefix (str, optional): Image prefix to generate full
            image path.
        test_mode (bool, optional): If set True, try...except will
            be turned off in __getitem__.
    """

    def __init__(self,
                 ann_file,
                 loader,
                 pipeline,
                 img_prefix='',
                 test_mode=False):
        super().__init__()
        self.test_mode = test_mode
        self.img_prefix = img_prefix
        self.ann_file = ann_file
        # load annotations
        loader.update(ann_file=ann_file)
        self.data_infos = build_loader(loader)
        # processing pipeline
        self.pipeline = Compose(pipeline)
        # set group flag and class, no meaning
        # for text detect and recognize
        self._set_group_flag()
        self.CLASSES = 0

    def __len__(self):
        return len(self.data_infos)

    def _set_group_flag(self):
        """Set flag."""
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix

    def prepare_train_img(self, index):
        """Get training data and annotations from pipeline.

        Args:
            index (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """
        img_info = self.data_infos[index]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, img_info):
        """Get testing data from pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """
        return self.prepare_train_img(img_info)

    def _log_error_index(self, index):
        """Logging data info of bad index."""
        try:
            data_info = self.data_infos[index]
            img_prefix = self.img_prefix
            print_log(f'Warning: skip broken file {data_info} '
                      f'with img_prefix {img_prefix}')
        except Exception as e:
            print_log(f'load index {index} with error {e}')

    def _get_next_index(self, index):
        """Get next index from dataset."""
        self._log_error_index(index)
        index = (index + 1) % len(self)
        return index

    def __getitem__(self, index):
        """Get training/test data from pipeline.

        Args:
            index (int): Index of data.

        Returns:
            dict: Training/test data.
        """
        if self.test_mode:
            return self.prepare_test_img(index)

        while True:
            try:
                data = self.prepare_train_img(index)
                if data is None:
                    raise Exception('prepared train data empty')
                break
            except Exception as e:
                print_log(f'prepare index {index} with error {e}')
                index = self._get_next_index(index)
        return data

    def format_results(self, results, **kwargs):
        """Placeholder to format result to dataset-specific output."""
        pass

    def evaluate(self, results, metric=None, logger=None, **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            dict[str: float]
        """
        raise NotImplementedError

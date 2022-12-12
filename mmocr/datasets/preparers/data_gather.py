# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import re
from typing import List, Optional, Sequence, Union

from mmocr.utils import list_files
from .data_preparer import DATA_GATHER


class BaseGatherer:
    """Base class for gather.

    Note: Gather assumes that all the annotation file is in the same directory
    and all the image files are in the same directory.

    Args:
        splits (list[str], optional): List of splits to gather. It' s
            the partition of the datasets. It's subset of
            ['train', 'val', 'test']. Defaults to None.
        img_root (str, optional): The directory of the images. Defaults to
            None.
        ann_root (str, optional): The directory of the annotation files.
            Defaults to None.
    """

    def __init__(self,
                 splits: Optional[List[str]] = None,
                 img_root: Optional[str] = None,
                 ann_root: Optional[str] = None) -> None:
        self.splits = splits
        self.img_root = img_root
        self.ann_root = ann_root

    def gather_single(self, split: str) -> Union[str, Sequence]:
        pass

    def __call__(self) -> Sequence:
        return list(map(self.gather_single, self.splits))


@DATA_GATHER.register_module()
class MonoGatherer(BaseGatherer):
    """Gather the dataset file. Specifically for the case that only one
    annotation file is needed. For example,

            img_001.jpg \
            img_002.jpg ---> train.json
            img_003.jpg /

    Args:
        train_ann_name (str, optional): The name of the annotation file for
            training. Defaults to None.
        val_ann_name (str, optional): The name of the annotation file for
            validation. Defaults to None.
        test_ann_name (str, optional): The name of the annotation file for
            testing. Defaults to None.
        train_img_postfix_dir (str, optional): The postfix of the image
            directory for training. The actual directory is the concatenation
            of img_root and train_img_postfix_dir. Defaults to 'train'.
        val_img_postfix_dir (str, optional): The postfix of the image
            directory for validation. The actual directory is the concatenation
            of img_root and val_img_postfix_dir. Defaults to 'val'. Defaults to
            'val'.
        test_img_postfix_dir (str, optional): The postfix of the image
            directory for testing. The actual directory is the concatenation
            of img_root and test_img_postfix_dir. Defaults to 'test'.
        splits (list[str], optional): List of splits to gather. It' s
            the partition of the datasets. It's subset of
            ['train', 'val', 'test']. Defaults to None.
        img_root (str, optional): The directory of the images. Defaults to
            None.
        ann_root (str, optional): The directory of the annotation files.
            Defaults to None.
    """

    def __init__(self,
                 train_ann_name: Optional[str] = None,
                 val_ann_name: Optional[str] = None,
                 test_ann_name: Optional[str] = None,
                 train_img_postfix_dir: str = 'train',
                 val_img_postfix_dir: str = 'val',
                 test_img_postfix_dir: str = 'test',
                 splits: Optional[List[str]] = None,
                 img_root: Optional[str] = None,
                 ann_root: Optional[str] = None) -> None:
        super().__init__(splits=splits, img_root=img_root, ann_root=ann_root)
        self.train_img_postfix_dir = train_img_postfix_dir
        self.val_img_postfix_dir = val_img_postfix_dir
        self.test_img_postfix_dir = test_img_postfix_dir
        self.train_ann_name = train_ann_name
        self.val_ann_name = val_ann_name
        self.test_ann_name = test_ann_name

    def gather_single(self, split) -> Union[str, Sequence]:
        """
        Args:
            split (str): The split to gather.

        Returns:
            str: Path to the annotation file.
        """
        ann_name = eval(f'self.{split}_ann_name')
        if ann_name is None:
            raise ValueError(f'{split}_ann must be specified in gather!')

        return (osp.join(self.img_root, eval(f'self.{split}_img_postfix_dir')),
                osp.join(self.ann_root, ann_name))


@DATA_GATHER.register_module()
class PairGatherer(BaseGatherer):
    """Gather the dataset files. Specifically for the paired annotations. That
    is to say, each image has a corresponding annotation file. For example,

            img_1.jpg <---> gt_img_1.txt
            img_2.jpg <---> gt_img_2.txt
            img_3.jpg <---> gt_img_3.txt

    Args:
        suffixes (List[str]): File suffixes that used for searching.
        rule (Sequence): The rule for pairing the files. The
                first element is the matching pattern for the file, and the
                second element is the replacement pattern, which should
                be a regular expression. For example, to map the image
                name img_1.jpg to the annotation name gt_img_1.txt,
                the rule is
                    [r'img_(\d+)\.([jJ][pP][gG])', r'gt_img_\1.txt'] # noqa: W605 E501
        train_img_postfix_dir (str, optional): The postfix directory of the
            training images. The actual directory of train images is img_root
            + train_img_postfix_dir. Defaults to 'train'.
        val_img_postfix_dir (str, optional): The postfix directory of the
            validation images. The actual directory of val images is img_root
            + val_img_postfix_dir. Defaults to 'val'.
        test_img_postfix_dir (str, optional): The postfix directory of the
            testing images. The actual directory of test images is img_root
            + test_img_postfix_dir. Defaults to 'test'.
        train_ann_postfix_dir (str, optional): The postfix directory of the
            training annotation files. The actual directory of train
            annotation files is ann_root + train_ann_postfix_dir. Defaults to
            'train'.
        val_ann_postfix_dir (str, optional): The postfix directory of the
            validation annotation files. The actual directory of val
            annotation files is ann_root + val_ann_postfix_dir. Defaults to
            'val'.
        splits (list[str], optional): List of splits to gather. It' s
            the partition of the datasets. It's subset of
            ['train', 'val', 'test']. Defaults to None.
    """

    def __init__(self,
                 img_suffixes: Optional[List[str]] = None,
                 rule: Optional[List[str]] = None,
                 train_img_postfix_dir: str = 'train',
                 val_img_postfix_dir: str = 'val',
                 test_img_postfix_dir: str = 'test',
                 train_ann_postfix_dir: str = 'train',
                 val_ann_postfix_dir: str = 'val',
                 test_ann_postfix_dir: str = 'test',
                 splits: Optional[List[str]] = None,
                 img_root: Optional[str] = None,
                 ann_root: Optional[str] = None) -> None:
        super().__init__(splits=splits, img_root=img_root, ann_root=ann_root)
        self.rule = rule
        self.img_suffixes = img_suffixes
        self.train_img_postfix_dir = train_img_postfix_dir
        self.val_img_postfix_dir = val_img_postfix_dir
        self.test_img_postfix_dir = test_img_postfix_dir
        self.train_ann_postfix_dir = train_ann_postfix_dir
        self.val_ann_postfix_dir = val_ann_postfix_dir
        self.test_ann_postfix_dir = test_ann_postfix_dir

    def gather_single(self, split) -> Union[str, Sequence]:
        """
        Args:
            split (str): The split to gather.

        Returns:
            List[Tuple]: A list of tuples (img_path, ann_path).
        """

        img_postfix_dir = eval(f'self.{split}_img_postfix_dir')
        ann_postfix_dir = eval(f'self.{split}_ann_postfix_dir')
        img_dir = osp.join(self.img_root, img_postfix_dir)
        ann_dir = osp.join(self.ann_root, ann_postfix_dir)
        img_list = list()
        ann_list = list()
        for img_path in list_files(img_dir, self.img_suffixes):
            ann_name = re.sub(self.rule[0], self.rule[1],
                              osp.basename(img_path))
            ann_path = osp.join(ann_dir, ann_name)
            img_list.append(img_path)
            ann_list.append(ann_path)

        return img_list, ann_list

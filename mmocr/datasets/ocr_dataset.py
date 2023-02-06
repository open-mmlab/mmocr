# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset import BaseDataset

from mmocr.registry import DATASETS


@DATASETS.register_module()
class OCRDataset(BaseDataset):
    r"""OCRDataset for text detection and text recognition.

    The annotation format is shown as follows.

    .. code-block:: none

        {
            "metainfo":
            {
              "dataset_type": "test_dataset",
              "task_name": "test_task"
            },
            "data_list":
            [
              {
                "img_path": "test_img.jpg",
                "height": 604,
                "width": 640,
                "instances":
                [
                  {
                    "bbox": [0, 0, 10, 20],
                    "bbox_label": 1,
                    "mask": [0,0,0,10,10,20,20,0],
                    "text": '123'
                  },
                  {
                    "bbox": [10, 10, 110, 120],
                    "bbox_label": 2,
                    "mask": [10,10],10,110,110,120,120,10]],
                    "extra_anns": '456'
                  }
                ]
              },
            ]
        }

    Args:
        ann_file (str): Annotation file path. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to None.
        data_prefix (dict): Prefix for training data. Defaults to
            dict(img_path='').
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
            load annotation file. ``OCRdataset`` can skip load annotations to
            save time by set ``lazy_init=False``. Defaults to False.
        max_refetch (int, optional): If ``OCRdataset.prepare_data`` get a
            None img. The maximum extra number of cycles to get a valid
            image. Defaults to 1000.

    Note:
        OCRDataset collects meta information from `annotation file` (the
        lowest priority), ``OCRDataset.METAINFO``(medium) and `metainfo
        parameter` (highest) passed to constructors. The lower priority meta
        information will be overwritten by higher one.

    Examples:
        Assume the annotation file is given above.
        >>> class CustomDataset(OCRDataset):
        >>>     METAINFO: dict = dict(task_name='custom_task',
        >>>                           dataset_type='custom_type')
        >>> metainfo=dict(task_name='custom_task_name')
        >>> custom_dataset = CustomDataset(
        >>>                      'path/to/ann_file',
        >>>                      metainfo=metainfo)
        >>> # meta information of annotation file will be overwritten by
        >>> # `CustomDataset.METAINFO`. The merged meta information will
        >>> # further be overwritten by argument `metainfo`.
        >>> custom_dataset.metainfo
        {'task_name': custom_task_name, dataset_type: custom_type}
    """

# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
from mmdet.datasets.api_wrappers import COCO
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset

import mmocr.utils as utils
from mmocr import digit_version
from mmocr.core.evaluation.hmean import eval_hmean


@DATASETS.register_module()
class IcdarDataset(CocoDataset):
    """Dataset for text detection while ann_file in coco format.

    Args:
        ann_file_backend (str): Storage backend for annotation file,
            should be one in ['disk', 'petrel', 'http']. Default to 'disk'.
    """
    CLASSES = ('text', )

    def __init__(self,
                 ann_file,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 select_first_k=-1,
                 ann_file_backend='disk'):
        # select first k images for fast debugging.
        self.select_first_k = select_first_k
        assert ann_file_backend in ['disk', 'petrel', 'http']
        self.ann_file_backend = ann_file_backend

        super().__init__(ann_file, pipeline, classes, data_root, img_prefix,
                         seg_prefix, proposal_file, test_mode, filter_empty_gt)

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """
        if self.ann_file_backend == 'disk':
            self.coco = COCO(ann_file)
        else:
            mmcv_version = digit_version(mmcv.__version__)
            if mmcv_version < digit_version('1.3.16'):
                raise Exception('Please update mmcv to 1.3.16 or higher '
                                'to enable "get_local_path" of "FileClient".')
            file_client = mmcv.FileClient(backend=self.ann_file_backend)
            with file_client.get_local_path(ann_file) as local_path:
                self.coco = COCO(local_path)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []

        count = 0
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            count = count + 1
            if count > self.select_first_k and self.select_first_k > 0:
                break
        return data_infos

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, masks_ignore, seg_map. "masks"  and
                "masks_ignore" are represented by polygon boundary
                point sequences.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ignore = []
        gt_masks_ann = []

        for ann in ann_info:
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
                gt_masks_ignore.append(ann.get(
                    'segmentation', None))  # to float32 for latter processing

            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks_ignore=gt_masks_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def evaluate(self,
                 results,
                 metric='hmean-iou',
                 logger=None,
                 score_thr=0.3,
                 rank_list=None,
                 **kwargs):
        """Evaluate the hmean metric.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            rank_list (str): json file used to save eval result
                of each image after ranking.
        Returns:
            dict[dict[str: float]]: The evaluation results.
        """
        assert utils.is_type_list(results, dict)

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['hmean-iou', 'hmean-ic13']
        metrics = set(metrics) & set(allowed_metrics)

        img_infos = []
        ann_infos = []
        for i in range(len(self)):
            img_info = {'filename': self.data_infos[i]['file_name']}
            img_infos.append(img_info)
            ann_infos.append(self.get_ann_info(i))

        eval_results = eval_hmean(
            results,
            img_infos,
            ann_infos,
            metrics=metrics,
            score_thr=score_thr,
            logger=logger,
            rank_list=rank_list)

        return eval_results

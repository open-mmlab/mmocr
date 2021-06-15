import numpy as np
from mmdet.datasets.builder import DATASETS

from mmocr.core.evaluation.hmean import eval_hmean
from mmocr.datasets.base_dataset import BaseDataset


@DATASETS.register_module()
class TextDetDataset(BaseDataset):

    def _parse_anno_info(self, annotations):
        """Parse bbox and mask annotation.
        Args:
            annotations (dict): Annotations of one image.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, masks_ignore. "masks"  and
                "masks_ignore" are represented by polygon boundary
                point sequences.
        """
        gt_bboxes, gt_bboxes_ignore = [], []
        gt_masks, gt_masks_ignore = [], []
        gt_labels = []
        for ann in annotations:
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(ann['bbox'])
                gt_masks_ignore.append(ann.get('segmentation', None))
            else:
                gt_bboxes.append(ann['bbox'])
                gt_labels.append(ann['category_id'])
                gt_masks.append(ann.get('segmentation', None))
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

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks_ignore=gt_masks_ignore,
            masks=gt_masks)

        return ann

    def prepare_train_img(self, index):
        """Get training data and annotations from pipeline.

        Args:
            index (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """
        img_ann_info = self.data_infos[index]
        img_info = {
            'filename': img_ann_info['file_name'],
            'height': img_ann_info['height'],
            'width': img_ann_info['width']
        }
        ann_info = self._parse_anno_info(img_ann_info['annotations'])
        results = dict(img_info=img_info, ann_info=ann_info)
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        self.pre_pipeline(results)

        return self.pipeline(results)

    def evaluate(self,
                 results,
                 metric='hmean-iou',
                 score_thr=0.3,
                 rank_list=None,
                 logger=None,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            score_thr (float): Score threshold for prediction map.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            rank_list (str): json file used to save eval result
                of each image after ranking.
        Returns:
            dict[str: float]
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['hmean-iou', 'hmean-ic13']
        metrics = set(metrics) & set(allowed_metrics)

        img_infos = []
        ann_infos = []
        for i in range(len(self)):
            img_ann_info = self.data_infos[i]
            img_info = {'filename': img_ann_info['file_name']}
            ann_info = self._parse_anno_info(img_ann_info['annotations'])
            img_infos.append(img_info)
            ann_infos.append(ann_info)

        eval_results = eval_hmean(
            results,
            img_infos,
            ann_infos,
            metrics=metrics,
            score_thr=score_thr,
            logger=logger,
            rank_list=rank_list)

        return eval_results

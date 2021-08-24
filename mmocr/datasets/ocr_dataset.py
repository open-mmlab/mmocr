# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.core.evaluation.ocr_metric import eval_ocr_metric
from mmocr.datasets.base_dataset import BaseDataset
from mmocr.datasets.builder import DATASETS


@DATASETS.register_module()
class OCRDataset(BaseDataset):

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['text'] = results['img_info']['text']

    def evaluate(self, results, metric='acc', logger=None, **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            dict[str: float]
        """
        gt_texts = []
        pred_texts = []
        for i in range(len(self)):
            item_info = self.data_infos[i]
            text = item_info['text']
            gt_texts.append(text)
            pred_texts.append(results[i]['text'])

        eval_results = eval_ocr_metric(pred_texts, gt_texts)

        return eval_results

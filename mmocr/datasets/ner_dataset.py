from mmdet.datasets.builder import DATASETS
from mmocr.core.evaluation.ner_metric import eval_ner
from mmocr.datasets.base_dataset import BaseDataset


@DATASETS.register_module()
class NerDataset(BaseDataset):
    """
    Args:
        ann_file (txt): Annotation file path.
        loader (dict): Dictionary to construct loader
            to load annotation infos.
        pipeline (list[dict]): Processing pipeline.
        img_prefix (str, optional): This parameter is not used.
        test_mode (bool, optional): If True, try...except will
            be turned off in __getitem__.
    """

    def __init__(self,
                 ann_file,
                 loader,
                 pipeline,
                 img_prefix='',
                 test_mode=False):
        super().__init__(
            ann_file, loader, pipeline, img_prefix='', test_mode=False)
        self.ann_file = ann_file

    def _parse_anno_info(self, ann):
        """Parse annotations of texts and labels for one text.

        Args:
            ann (dict): Annotations of texts and labels for one text
        Returns:
            ans: A dict.
        """
        text = ann['text']
        label = ann['label']

        return {'text': text, 'label': label}

    def prepare_train_img(self, index):
        """Get training data and annotations after pipeline.

        Args:
            index (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        img_ann_info = self.data_infos[index]
        ann_info = self._parse_anno_info(img_ann_info)
        results = dict(ann_info)

        return self.pipeline(results)

    def evaluate(self, results, metric=None, logger=None, **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            info (dict): A dict containing the following keys:
             'acc', 'recall', 'f1'.
        """
        gt_infos = []
        for i in range(len(self)):
            gt_infos.append(self.data_infos[i])
        info = eval_ner(results, gt_infos)
        return info

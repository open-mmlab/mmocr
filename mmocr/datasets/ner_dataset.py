from mmdet.datasets.builder import DATASETS
from mmocr.core.evaluation.ner_metric import eval_ner_f1
from mmocr.datasets.base_dataset import BaseDataset


@DATASETS.register_module()
class NerDataset(BaseDataset):
    """Custom dataset for named entity recognition tasks.

    Args:
        ann_file (txt): Annotation file path.
        loader (dict): Dictionary to construct loader
            to load annotation infos.
        pipeline (list[dict]): Processing pipeline.
        test_mode (bool, optional): If True, try...except will
            be turned off in __getitem__.
    """
    def prepare_train_img(self, index):
        """Get training data and annotations after pipeline.

        Args:
            index (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        ann_info = self.data_infos[index]

        return self.pipeline(ann_info)

    def evaluate(self, results, metric=None, logger=None, **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            info (dict): A dict containing the following keys:
             'acc', 'recall', 'f1-score'.
        """
        gt_infos = list(self.data_infos)
        eval_results = eval_ner_f1(results, gt_infos)
        return eval_results

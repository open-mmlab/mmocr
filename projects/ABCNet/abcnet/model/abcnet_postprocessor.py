# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.models.textdet.postprocessors.base import BaseTextDetPostProcessor
from mmocr.registry import MODELS
from ..utils import bezier2poly


@MODELS.register_module()
class ABCNetPostprocessor(BaseTextDetPostProcessor):
    """Post-processing methods for ABCNet.

    Args:
        num_classes (int): Number of classes.
        use_sigmoid_cls (bool): Whether to use sigmoid for classification.
        strides (tuple): Strides of each feature map.
        norm_by_strides (bool): Whether to normalize the regression targets by
            the strides.
        bbox_coder (dict): Config dict for bbox coder.
        text_repr_type (str): Text representation type, 'poly' or 'quad'.
        with_bezier (bool): Whether to use bezier curve for text detection.
        train_cfg (dict): Config dict for training.
        test_cfg (dict): Config dict for testing.
    """

    def __init__(
        self,
        text_repr_type='poly',
        rescale_fields=['beziers', 'polygons'],
    ):
        super().__init__(
            text_repr_type=text_repr_type, rescale_fields=rescale_fields)

    def merge_predict(self, spotting_data_samples, recog_data_samples):
        texts = [ds.pred_text.item for ds in recog_data_samples]
        start = 0
        for spotting_data_sample in spotting_data_samples:
            end = start + len(spotting_data_sample.pred_instances)
            spotting_data_sample.pred_instances.texts = texts[start:end]
            start = end
        return spotting_data_samples

    # TODO: fix docstr
    def __call__(self,
                 spotting_data_samples,
                 recog_data_samples,
                 training: bool = False):
        """Postprocess pred_results according to metainfos in data_samples.

        Args:
            pred_results (Union[Tensor, List[Tensor]]): The prediction results
                stored in a tensor or a list of tensor. Usually each item to
                be post-processed is expected to be a batched tensor.
            data_samples (list[TextDetDataSample]): Batch of data_samples,
                each corresponding to a prediction result.
            training (bool): Whether the model is in training mode. Defaults to
                False.

        Returns:
            list[TextDetDataSample]: Batch of post-processed datasamples.
        """
        spotting_data_samples = list(
            map(self._process_single, spotting_data_samples))
        return self.merge_predict(spotting_data_samples, recog_data_samples)

    def _process_single(self, data_sample):
        """Process prediction results from one image.

        Args:
            pred_result (Union[Tensor, List[Tensor]]): Prediction results of an
                image.
            data_sample (TextDetDataSample): Datasample of an image.
        """
        data_sample = self.get_text_instances(data_sample)
        if self.rescale_fields and len(self.rescale_fields) > 0:
            assert isinstance(self.rescale_fields, list)
            assert set(self.rescale_fields).issubset(
                set(data_sample.pred_instances.keys()))
            data_sample = self.rescale(data_sample, data_sample.scale_factor)
        return data_sample

    def get_text_instances(self, data_sample, **kwargs):
        """Get text instance predictions of one image.

        Args:
            pred_result (tuple(Tensor)): Prediction results of an image.
            data_sample (TextDetDataSample): Datasample of an image.
            **kwargs: Other parameters. Configurable via ``__init__.train_cfg``
                and ``__init__.test_cfg``.

        Returns:
            TextDetDataSample: A new DataSample with predictions filled in.
            The polygon/bbox results are usually saved in
            ``TextDetDataSample.pred_instances.polygons`` or
            ``TextDetDataSample.pred_instances.bboxes``. The confidence scores
            are saved in ``TextDetDataSample.pred_instances.scores``.
        """
        data_sample = data_sample.cpu().numpy()
        pred_instances = data_sample.pred_instances
        data_sample.pred_instances.polygons = list(
            map(bezier2poly, pred_instances.beziers))
        return data_sample

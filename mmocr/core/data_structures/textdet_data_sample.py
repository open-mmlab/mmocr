# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.data import BaseDataElement, InstanceData


class TextDetDataSample(BaseDataElement):
    """A data structure interface of MMOCR. They are used as interfaces between
    different components.

    The attributes in ``TextDetDataSample`` are divided into two parts:

        - ``gt_instances``(InstanceData): Ground truth of instance annotations.
        - ``pred_instances``(InstanceData): Instances of model predictions.

    Examples:
         >>> import torch
         >>> import numpy as np
         >>> from mmengine.data import InstanceData
         >>> from mmocr.core import TextDetDataSample
         >>> # gt_instances
         >>> data_sample = TextDetDataSample()
         >>> img_meta = dict(img_shape=(800, 1196, 3),
         ...                 pad_shape=(800, 1216, 3))
         >>> gt_instances = InstanceData(metainfo=img_meta)
         >>> gt_instances.bboxes = torch.rand((5, 4))
         >>> gt_instances.labels = torch.rand((5,))
         >>> data_sample.gt_instances = gt_instances
         >>> assert 'img_shape' in data_sample.gt_instances.metainfo_keys()
         >>> len(data_sample.gt_instances)
         5
         >>> print(data_sample)
        <TextDetDataSample(
            META INFORMATION
            DATA FIELDS
            gt_instances: <InstanceData(
                    META INFORMATION
                    pad_shape: (800, 1216, 3)
                    img_shape: (800, 1196, 3)
                    DATA FIELDS
                    labels: tensor([0.8533, 0.1550, 0.5433, 0.7294, 0.5098])
                    bboxes:
                    tensor([[9.7725e-01, 5.8417e-01, 1.7269e-01, 6.5694e-01],
                            [1.7894e-01, 5.1780e-01, 7.0590e-01, 4.8589e-01],
                            [7.0392e-01, 6.6770e-01, 1.7520e-01, 1.4267e-01],
                            [2.2411e-01, 5.1962e-01, 9.6953e-01, 6.6994e-01],
                            [4.1338e-01, 2.1165e-01, 2.7239e-04, 6.8477e-01]])
                ) at 0x7f21fb1b9190>
        ) at 0x7f21fb1b9880>
         >>> # pred_instances
         >>> pred_instances = InstanceData(metainfo=img_meta)
         >>> pred_instances.bboxes = torch.rand((5, 4))
         >>> pred_instances.scores = torch.rand((5,))
         >>> data_sample = TextDetDataSample(pred_instances=pred_instances)
         >>> assert 'pred_instances' in data_sample
         >>> data_sample = TextDetDataSample()
         >>> gt_instances_data = dict(
         ...                        bboxes=torch.rand(2, 4),
         ...                        labels=torch.rand(2),
         ...                        masks=np.random.rand(2, 2, 2))
         >>> gt_instances = InstanceData(**gt_instances_data)
         >>> data_sample.gt_instances = gt_instances
         >>> assert 'gt_instances' in data_sample
         >>> assert 'masks' in data_sample.gt_instances
    """

    @property
    def gt_instances(self) -> InstanceData:
        """InstanceData: groundtruth instances."""
        return self._gt_instances

    @gt_instances.setter
    def gt_instances(self, value: InstanceData):
        """gt_instances setter."""
        self.set_field(value, '_gt_instances', dtype=InstanceData)

    @gt_instances.deleter
    def gt_instances(self):
        """gt_instances deleter."""
        del self._gt_instances

    @property
    def pred_instances(self) -> InstanceData:
        """InstanceData: prediction instances."""
        return self._pred_instances

    @pred_instances.setter
    def pred_instances(self, value: InstanceData):
        """pred_instances setter."""
        self.set_field(value, '_pred_instances', dtype=InstanceData)

    @pred_instances.deleter
    def pred_instances(self):
        """pred_instances deleter."""
        del self._pred_instances

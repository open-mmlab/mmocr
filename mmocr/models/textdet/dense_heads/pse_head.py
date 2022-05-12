# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.registry import MODELS
from . import PANHead


@MODELS.register_module()
class PSEHead(PANHead):
    """The class for PSENet head.

    Args:
        in_channels (list[int]): A list of 4 numbers of input channels.
        out_channels (int): Number of output channels.
        downsample_ratio (float): Downsample ratio.
        loss (dict): Configuration dictionary for loss type. Supported loss
            types are "PANLoss" and "PSELoss".
        postprocessor (dict): Config of postprocessor for PSENet.
        train_cfg, test_cfg (dict): Depreciated.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 downsample_ratio=0.25,
                 loss=dict(type='PSELoss'),
                 postprocessor=dict(
                     type='PSEPostprocessor', text_repr_type='poly'),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            downsample_ratio=downsample_ratio,
            loss=loss,
            postprocessor=postprocessor,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            **kwargs)

import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet.models.builder import HEADS, build_loss

from . import HeadMixin


@HEADS.register_module()
class TextSnakeHead(HeadMixin, BaseModule):
    """The class for TextSnake head: TextSnake: A Flexible Representation for
    Detecting Text of Arbitrary Shapes.

    [https://arxiv.org/abs/1807.01544]
    """

    def __init__(self,
                 in_channels,
                 decoding_type='textsnake',
                 text_repr_type='poly',
                 loss=dict(type='TextSnakeLoss'),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(
                     type='Normal', layer='Conv2d', mean=0, std=0.01)):
        super().__init__(init_cfg=init_cfg)

        assert isinstance(in_channels, int)
        self.in_channels = in_channels
        self.out_channels = 5
        self.downsample_ratio = 1.0
        self.decoding_type = decoding_type
        self.text_repr_type = text_repr_type
        self.loss_module = build_loss(loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.out_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0)
        # self.init_weights()

    '''
    def init_weights(self):
        normal_init(self.out_conv, mean=0, std=0.01)
    '''

    def forward(self, inputs):
        outputs = self.out_conv(inputs)
        return outputs

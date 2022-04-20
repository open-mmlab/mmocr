# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, Sequential

from mmocr.models.builder import HEADS
from .head_mixin import HeadMixin


@HEADS.register_module()
class EASTHead(HeadMixin, BaseModule):

    def __init__(self,
                 box_type,
                 num_classes,
                 in_channels,
                 loss=dict(type='EASTLoss', box_type='RBOX'),
                 postprocessor=dict(
                     type='EASTPostprocessor',
                     text_repr_type='quad',
                     box_type='RBOX'),
                 init_cfg=[
                     dict(type='Kaiming', layer='Conv'),
                     dict(
                         type='Constant', layer='BatchNorm', val=1., bias=1e-4)
                 ],
                 train_cfg=None,
                 test_cfg=None):
        BaseModule.__init__(self, init_cfg=init_cfg)
        HeadMixin.__init__(self, loss, postprocessor)
        assert box_type in ['RBOX', 'QUAD']
        assert box_type == loss['box_type'] == postprocessor['box_type']
        self.box_type = box_type
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self._init_cls_convs()
        self._init_reg_convs()

    def _init_reg_convs(self):
        """Initialize bbox regression conv layers of the head."""
        if self.box_type == 'RBOX':
            self.ang_conv = Sequential(
                ConvModule(
                    self.in_channels,
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='ReLU')))

        out_channels = 4 if self.box_type == 'RBOX' else 8
        self.reg_conv = ConvModule(
            self.in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))

    def _init_cls_convs(self):
        """Initialize score prediction conv layers of the head."""
        self.score_conv = Sequential(
            ConvModule(
                self.in_channels,
                out_channels=1,
                kernel_size=1,
                stride=1,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU')))

    def forward(self, feat):
        score_res = self.score_conv(feat)
        reg_res = self.reg_conv(feat)
        if self.box_type == 'RBOX':
            ang_res = self.ang_conv(feat)
            preds = {
                'cls_pred': score_res,
                'bbox_pred': reg_res,
                'ang_pred': ang_res
            }
        else:
            preds = {'cls_pred': score_res, 'bbox_pred': reg_res}
        return preds

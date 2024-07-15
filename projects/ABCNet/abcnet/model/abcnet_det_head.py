# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule, Scale
from mmdet.models.utils import multi_apply

from mmocr.models.textdet.heads.base import BaseTextDetHead
from mmocr.registry import MODELS

INF = 1e8


@MODELS.register_module()
class ABCNetDetHead(BaseTextDetHead):

    def __init__(self,
                 in_channels,
                 module_loss=dict(type='ABCNetDetModuleLoss'),
                 postprocessor=dict(type='ABCNetDetPostprocessor'),
                 num_classes=1,
                 strides=(4, 8, 16, 32, 64),
                 feat_channels=256,
                 stacked_convs=4,
                 dcn_on_last_conv=False,
                 conv_bias='auto',
                 norm_on_bbox=False,
                 centerness_on_reg=False,
                 use_sigmoid_cls=True,
                 with_bezier=False,
                 use_scale=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01))):
        super().__init__(
            module_loss=module_loss,
            postprocessor=postprocessor,
            init_cfg=init_cfg)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.strides = strides
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.dcn_on_last_conv = dcn_on_last_conv
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_bezier = with_bezier
        self.use_scale = use_scale
        self.use_sigmoid_cls = use_sigmoid_cls
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self._init_cls_convs()
        self._init_reg_convs()
        self._init_predictor()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        # if self.use_scale:
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def _init_cls_convs(self):
        """Initialize classification conv layers of the head."""
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

    def _init_reg_convs(self):
        """Initialize bbox regression conv layers of the head."""
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        self.conv_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        if self.with_bezier:
            self.conv_bezier = nn.Conv2d(
                self.feat_channels, 16, kernel_size=3, stride=1, padding=1)

    def forward(self, feats, data_samples=None):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level, \
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
                centernesses (list[Tensor]): centerness for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
        """

        return multi_apply(self.forward_single, feats[1:], self.scales,
                           self.strides)

    def forward_single(self, x, scale, stride):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions and centerness \
                predictions of input feature maps. If ``with_bezier`` is True,
                Bezier prediction will also be returned.
        """
        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.conv_cls(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = self.conv_reg(reg_feat)
        if self.with_bezier:
            bezier_pred = self.conv_bezier(reg_feat)

        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        if self.use_scale:
            bbox_pred = scale(bbox_pred).float()
        # else:
        #     bbox_pred = bbox_pred.float()
        if self.norm_on_bbox:
            # bbox_pred needed for gradient computation has been modified
            # by F.relu(bbox_pred) when run with PyTorch 1.10. So replace
            # F.relu(bbox_pred) with bbox_pred.clamp(min=0)
            bbox_pred = bbox_pred.clamp(min=0)
        else:
            bbox_pred = bbox_pred.exp()

        if self.with_bezier:
            return cls_score, bbox_pred, centerness, bezier_pred
        else:
            return cls_score, bbox_pred, centerness

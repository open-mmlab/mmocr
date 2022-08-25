# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch
from mmdet.structures import DetDataSample
from mmdet.testing import demo_mm_inputs
from mmengine.config import Config
from mmengine.structures import InstanceData

from mmocr.registry import MODELS
from mmocr.structures import TextDetDataSample


class TestMMDetWrapper(unittest.TestCase):

    def setUp(self):
        model_cfg_fcos = dict(
            type='MMDetWrapper',
            cfg=dict(
                type='FCOS',
                data_preprocessor=dict(
                    type='DetDataPreprocessor',
                    mean=[102.9801, 115.9465, 122.7717],
                    std=[1.0, 1.0, 1.0],
                    bgr_to_rgb=False,
                    pad_size_divisor=32),
                backbone=dict(
                    type='ResNet',
                    depth=50,
                    num_stages=4,
                    out_indices=(0, 1, 2, 3),
                    frozen_stages=1,
                    norm_cfg=dict(type='BN', requires_grad=False),
                    norm_eval=True,
                    style='caffe',
                    init_cfg=dict(
                        type='Pretrained',
                        checkpoint='open-mmlab://detectron/resnet50_caffe')),
                neck=dict(
                    type='FPN',
                    in_channels=[256, 512, 1024, 2048],
                    out_channels=256,
                    start_level=1,
                    add_extra_convs='on_output',  # use P5
                    num_outs=5,
                    relu_before_extra_convs=True),
                bbox_head=dict(
                    type='FCOSHead',
                    num_classes=2,
                    in_channels=256,
                    stacked_convs=4,
                    feat_channels=256,
                    strides=[8, 16, 32, 64, 128],
                    loss_cls=dict(
                        type='FocalLoss',
                        use_sigmoid=True,
                        gamma=2.0,
                        alpha=0.25,
                        loss_weight=1.0),
                    loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                    loss_centerness=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=True,
                        loss_weight=1.0)),
                # testing settings
                test_cfg=dict(
                    nms_pre=1000,
                    min_bbox_size=0,
                    score_thr=0.05,
                    nms=dict(type='nms', iou_threshold=0.5),
                    max_per_img=100)))
        model_cfg_maskrcnn = dict(
            type='MMDetWrapper',
            text_repr_type='quad',
            cfg=dict(
                type='MaskRCNN',
                data_preprocessor=dict(
                    type='DetDataPreprocessor',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    bgr_to_rgb=True,
                    pad_size_divisor=32),
                backbone=dict(
                    type='ResNet',
                    depth=50,
                    num_stages=4,
                    out_indices=(0, 1, 2, 3),
                    frozen_stages=1,
                    norm_cfg=dict(type='BN', requires_grad=True),
                    norm_eval=True,
                    style='pytorch',
                    init_cfg=dict(
                        type='Pretrained',
                        checkpoint='torchvision://resnet50')),
                neck=dict(
                    type='FPN',
                    in_channels=[256, 512, 1024, 2048],
                    out_channels=256,
                    num_outs=5),
                rpn_head=dict(
                    type='RPNHead',
                    in_channels=256,
                    feat_channels=256,
                    anchor_generator=dict(
                        type='AnchorGenerator',
                        scales=[8],
                        ratios=[0.5, 1.0, 2.0],
                        strides=[4, 8, 16, 32, 64]),
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=[.0, .0, .0, .0],
                        target_stds=[1.0, 1.0, 1.0, 1.0]),
                    loss_cls=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=True,
                        loss_weight=1.0),
                    loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
                roi_head=dict(
                    type='StandardRoIHead',
                    bbox_roi_extractor=dict(
                        type='SingleRoIExtractor',
                        roi_layer=dict(
                            type='RoIAlign', output_size=7, sampling_ratio=0),
                        out_channels=256,
                        featmap_strides=[4, 8, 16, 32]),
                    bbox_head=dict(
                        type='Shared2FCBBoxHead',
                        in_channels=256,
                        fc_out_channels=1024,
                        roi_feat_size=7,
                        num_classes=80,
                        bbox_coder=dict(
                            type='DeltaXYWHBBoxCoder',
                            target_means=[0., 0., 0., 0.],
                            target_stds=[0.1, 0.1, 0.2, 0.2]),
                        reg_class_agnostic=False,
                        loss_cls=dict(
                            type='CrossEntropyLoss',
                            use_sigmoid=False,
                            loss_weight=1.0),
                        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
                    mask_roi_extractor=dict(
                        type='SingleRoIExtractor',
                        roi_layer=dict(
                            type='RoIAlign', output_size=14, sampling_ratio=0),
                        out_channels=256,
                        featmap_strides=[4, 8, 16, 32]),
                    mask_head=dict(
                        type='FCNMaskHead',
                        num_convs=4,
                        in_channels=256,
                        conv_out_channels=256,
                        num_classes=80,
                        loss_mask=dict(
                            type='CrossEntropyLoss',
                            use_mask=True,
                            loss_weight=1.0))),
                # model training and testing settings
                train_cfg=dict(
                    rpn=dict(
                        assigner=dict(
                            type='MaxIoUAssigner',
                            pos_iou_thr=0.7,
                            neg_iou_thr=0.3,
                            min_pos_iou=0.3,
                            match_low_quality=True,
                            ignore_iof_thr=-1),
                        sampler=dict(
                            type='RandomSampler',
                            num=256,
                            pos_fraction=0.5,
                            neg_pos_ub=-1,
                            add_gt_as_proposals=False),
                        allowed_border=-1,
                        pos_weight=-1,
                        debug=False),
                    rpn_proposal=dict(
                        nms_pre=2000,
                        max_per_img=1000,
                        nms=dict(type='nms', iou_threshold=0.7),
                        min_bbox_size=0),
                    rcnn=dict(
                        assigner=dict(
                            type='MaxIoUAssigner',
                            pos_iou_thr=0.5,
                            neg_iou_thr=0.5,
                            min_pos_iou=0.5,
                            match_low_quality=True,
                            ignore_iof_thr=-1),
                        sampler=dict(
                            type='RandomSampler',
                            num=512,
                            pos_fraction=0.25,
                            neg_pos_ub=-1,
                            add_gt_as_proposals=True),
                        mask_size=28,
                        pos_weight=-1,
                        debug=False)),
                test_cfg=dict(
                    rpn=dict(
                        nms_pre=1000,
                        max_per_img=1000,
                        nms=dict(type='nms', iou_threshold=0.7),
                        min_bbox_size=0),
                    rcnn=dict(
                        score_thr=0.05,
                        nms=dict(type='nms', iou_threshold=0.5),
                        max_per_img=100,
                        mask_thr_binary=0.5))))

        self.FCOS = MODELS.build(Config(model_cfg_fcos))
        self.MRCNN = MODELS.build(Config(model_cfg_maskrcnn))

    def test_one_stage_wrapper(self):
        packed_inputs = demo_mm_inputs(
            2, [[3, 128, 128], [3, 128, 128]], num_classes=2)
        # Test forward train
        bi, ds = self.FCOS.data_preprocessor(packed_inputs, True)
        losses = self.FCOS.forward(bi, ds, mode='loss')
        assert isinstance(losses, dict)
        # Test forward test
        self.FCOS.eval()
        with torch.no_grad():
            batch_results = self.FCOS.forward(bi, ds, mode='predict')
            self.assertEqual(len(batch_results), 2)
            self.assertIsInstance(batch_results[0], TextDetDataSample)

    def test_mask_two_stage_wrapper(self):
        packed_inputs = demo_mm_inputs(
            2, [[3, 128, 128], [3, 128, 128]], num_classes=2, with_mask=True)
        # Test forward train
        bi, ds = self.MRCNN.data_preprocessor(packed_inputs, True)
        losses = self.MRCNN.forward(bi, ds, mode='loss')
        assert isinstance(losses, dict)
        # Test forward test
        self.MRCNN.eval()
        with torch.no_grad():
            batch_results = self.MRCNN.forward(bi, ds, mode='predict')
            self.assertEqual(len(batch_results), 2)
            self.assertIsInstance(batch_results[0], TextDetDataSample)

    def test_adapt_predictions(self):
        data_sample = DetDataSample()
        pred_instances = InstanceData()
        pred_instances.scores = torch.randn(1)
        pred_instances.labels = torch.Tensor([1])
        pred_instances.bboxes = torch.Tensor([[0, 0, 2, 2]])
        pred_instances.masks = torch.rand(1, 10, 10)
        data_sample.pred_instances = pred_instances
        results = self.MRCNN.adapt_predictions([data_sample])
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], TextDetDataSample)
        self.assertTrue('polygons' in results[0].pred_instances.keys())

        data_sample = DetDataSample()
        pred_instances = InstanceData()
        pred_instances.scores = torch.randn(1)
        pred_instances.labels = torch.Tensor([1])
        pred_instances.bboxes = torch.Tensor([[0, 0, 2, 2]])
        data_sample.pred_instances = pred_instances
        results = self.FCOS.adapt_predictions([data_sample])
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], TextDetDataSample)
        self.assertTrue('polygons' in results[0].pred_instances.keys())

# Copyright (c) OpenMMLab. All rights reserved.
model_poly = dict(
    type='PANet',
    backbone=dict(
        type='mmdet.ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
        norm_eval=True,
        style='caffe'),
    neck=dict(type='FPEM_FFM', in_channels=[64, 128, 256, 512]),
    bbox_head=dict(
        type='PANHead',
        in_channels=[128, 128, 128, 128],
        out_channels=6,
        loss=dict(type='PANLoss'),
        postprocessor=dict(type='PANPostprocessor', text_repr_type='poly')),
    train_cfg=None,
    test_cfg=None)

model_quad = dict(
    type='PANet',
    backbone=dict(
        type='mmdet.ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
        norm_eval=True,
        style='caffe'),
    neck=dict(type='FPEM_FFM', in_channels=[64, 128, 256, 512]),
    bbox_head=dict(
        type='PANHead',
        in_channels=[128, 128, 128, 128],
        out_channels=6,
        loss=dict(type='PANLoss'),
        postprocessor=dict(type='PANPostprocessor', text_repr_type='quad')),
    train_cfg=None,
    test_cfg=None)

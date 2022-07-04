model = dict(
    type='PANet',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='caffe'),
    neck=dict(type='FPEM_FFM', in_channels=[256, 512, 1024, 2048]),
    bbox_head=dict(
        type='PANHead',
        in_channels=[128, 128, 128, 128],
        out_channels=6,
        loss_module=dict(type='PANLoss', speedup_bbox_thr=32),
        postprocessor=dict(type='PANPostprocessor', text_repr_type='poly')),
    train_cfg=None,
    test_cfg=None)

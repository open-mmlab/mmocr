preprocess_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
    pad_size_divisor=32)

model = dict(
    type='DRRG',
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        norm_eval=True,
        style='caffe'),
    neck=dict(
        type='FPN_UNet', in_channels=[256, 512, 1024, 2048], out_channels=32),
    det_head=dict(
        type='DRRGHead',
        in_channels=32,
        text_region_thr=0.3,
        center_region_thr=0.4,
        loss_module=dict(type='DRRGLoss'),
        postprocessor=dict(type='DRRGPostprocessor', link_thr=0.80)))

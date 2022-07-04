preprocess_cfg = dict(
    mean=[122.67891434, 116.66876762, 104.00698793],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
    pad_size_divisor=32)

model = dict(
    type='DBNet',
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        stage_with_dcn=(False, True, True, True)),
    neck=dict(
        type='FPNC', in_channels=[256, 512, 1024, 2048], lateral_channels=256),
    det_head=dict(
        type='DBHead',
        in_channels=256,
        loss_module=dict(type='DBLoss'),
        postprocessor=dict(type='DBPostprocessor', text_repr_type='quad')),
    preprocess_cfg=preprocess_cfg)

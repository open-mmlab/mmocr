# Copyright (c) OpenMMLab. All rights reserved.
model = dict(
    type='FCENet',
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        norm_eval=False,
        style='pytorch'),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        add_extra_convs='on_output',
        num_outs=3,
        relu_before_extra_convs=True,
        act_cfg=None),
    bbox_head=dict(
        type='FCEHead',
        in_channels=256,
        scales=(8, 16, 32),
        fourier_degree=5,
        loss=dict(type='FCELoss', num_sample=50),
        postprocessor=dict(
            type='FCEPostprocessor',
            text_repr_type='quad',
            num_reconstr_points=50,
            alpha=1.2,
            beta=1.0,
            score_thr=0.3)))

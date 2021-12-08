model = dict(
    type='TextSnake',
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
    bbox_head=dict(
        type='TextSnakeHead',
        in_channels=32,
        text_repr_type='poly',
        loss=dict(type='TextSnakeLoss')),
    train_cfg=None,
    test_cfg=None)

model = dict(
    type='DBNet',
    backbone=dict(
        type='mmdet.ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
        norm_eval=False,
        style='caffe'),
    neck=dict(
        type='FPNC',
        in_channels=[64, 128, 256, 512],
        lateral_channels=256,
        asf_cfg=dict(attention_type='ScaleChannelSpatial')),
    bbox_head=dict(
        type='DBHead',
        in_channels=256,
        loss=dict(type='DBLoss', alpha=5.0, beta=10.0, bbce_loss=True),
        postprocessor=dict(
            type='DBPostprocessor', text_repr_type='quad',
            epsilon_ratio=0.002)),
    train_cfg=None,
    test_cfg=None)

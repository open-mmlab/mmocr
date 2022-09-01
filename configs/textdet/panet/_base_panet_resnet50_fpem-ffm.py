_base_ = '_base_panet_resnet18_fpem-ffm.py'

model = dict(
    type='PANet',
    backbone=dict(
        _delete_=True,
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    neck=dict(in_channels=[256, 512, 1024, 2048]),
    det_head=dict(postprocessor=dict(text_repr_type='poly')))

_base_ = [
    '../../_base_/models/ocr_mask_rcnn_r50_fpn_ohem_poly.py',
    '../../_base_/schedules/schedule_160e.py', '../../_base_/runtime_10e.py'
]

dataset_type = 'IcdarDataset'
data_root = 'data/ctw1500/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(
        type='mmdet.LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='ScaleAspectJitter',
        img_scale=None,
        keep_ratio=False,
        resize_type='indep_sample_in_range',
        scale_range=(640, 2560)),
    dict(type='mmdet.RandomFlip', flip_ratio=0.5),
    dict(type='mmdet.Normalize', **img_norm_cfg),
    dict(
        type='RandomCropInstances',
        target_size=(640, 640),
        mask_type='union_all',
        instance_key='gt_masks'),
    dict(type='mmdet.Pad', size_divisor=32),
    dict(type='mmdet.DefaultFormatBundle'),
    dict(
        type='mmdet.Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(
        type='mmdet.LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='mmdet.MultiScaleFlipAug',
        # resize the long size to 1600
        img_scale=(1600, 1600),
        flip=False,
        transforms=[
            dict(type='mmdet.Resize', keep_ratio=True),
            # no flip
            dict(type='mmdet.RandomFlip'),
            dict(type='mmdet.Normalize', **img_norm_cfg),
            dict(type='mmdet.ImageToTensor', keys=['img']),
            dict(type='mmdet.Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=data_root + '/instances_training.json',
        img_prefix=data_root + '/imgs',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        # select_first_k=1,
        ann_file=data_root + '/instances_test.json',
        img_prefix=data_root + '/imgs',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # select_first_k=1,
        ann_file=data_root + '/instances_test.json',
        img_prefix=data_root + '/imgs',
        pipeline=test_pipeline))
evaluation = dict(interval=10, metric='hmean-iou')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_cfg = None
test_cfg = None

train_pipeline = [
    dict(
        type='mmdet.LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='ColorJitter', brightness=32.0 / 255, saturation=0.5),
    dict(type='mmdet.Normalize', **img_norm_cfg),
    dict(
        type='ScaleAspectJitter',
        img_scale=[(3000, 640)],
        ratio_range=(0.7, 1.3),
        aspect_ratio_range=(0.9, 1.1),
        multiscale_mode='value',
        keep_ratio=False),
    # shrink_ratio is from big to small. The 1st must be 1.0
    dict(type='PANetTargets', shrink_ratio=(1.0, 0.7)),
    dict(type='mmdet.RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='RandomRotateTextDet'),
    dict(
        type='RandomCropInstances',
        target_size=(640, 640),
        instance_key='gt_kernels'),
    dict(type='mmdet.Pad', size_divisor=32),
    dict(
        type='CustomFormatBundle',
        keys=['gt_kernels', 'gt_mask'],
        visualize=dict(flag=False, boundary_key='gt_kernels')),
    dict(type='mmdet.Collect', keys=['img', 'gt_kernels', 'gt_mask'])
]
test_pipeline = [
    dict(
        type='mmdet.LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='mmdet.MultiScaleFlipAug',
        img_scale=(3000, 640),
        flip=False,
        transforms=[
            dict(type='mmdet.Resize', img_scale=(3000, 640), keep_ratio=True),
            dict(type='mmdet.Normalize', **img_norm_cfg),
            dict(type='mmdet.Pad', size_divisor=32),
            dict(type='mmdet.ImageToTensor', keys=['img']),
            dict(type='mmdet.Collect', keys=['img']),
        ])
]

dataset_type = 'TextDetDataset'
img_prefix = 'tests/data/toy_dataset/imgs'
train_anno_file = 'tests/data/toy_dataset/instances_test.txt'
train1 = dict(
    type=dataset_type,
    img_prefix=img_prefix,
    ann_file=train_anno_file,
    loader=dict(
        type='HardDiskLoader',
        repeat=4,
        parser=dict(
            type='LineJsonParser',
            keys=['file_name', 'height', 'width', 'annotations'])),
    pipeline=train_pipeline,
    test_mode=False)

data_root = 'tests/data/toy_dataset'
train2 = dict(
    type='IcdarDataset',
    ann_file=data_root + '/instances_test.json',
    img_prefix=data_root + '/imgs',
    pipeline=train_pipeline)

test_anno_file = 'tests/data/toy_dataset/instances_test.txt'
test = dict(
    type=dataset_type,
    img_prefix=img_prefix,
    ann_file=test_anno_file,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineJsonParser',
            keys=['file_name', 'height', 'width', 'annotations'])),
    pipeline=test_pipeline,
    test_mode=True)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(type='ConcatDataset', datasets=[train1, train2]),
    val=dict(type='ConcatDataset', datasets=[test]),
    test=dict(type='ConcatDataset', datasets=[test]))

evaluation = dict(interval=1, metric='hmean-iou')

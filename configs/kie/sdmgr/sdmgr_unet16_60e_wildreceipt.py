dataset_type = 'KIEDataset'
data_root = 'data/wildreceipt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
max_scale, min_scale = 1024, 512

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(max_scale, min_scale), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='KIEFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'relations', 'texts', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(max_scale, min_scale), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='KIEFormatBundle'),
    dict(type='Collect', keys=['img', 'relations', 'texts', 'gt_bboxes'])
]

vocab_file = 'dict.txt'
class_file = 'class_list.txt'

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file='train.txt',
        pipeline=train_pipeline,
        data_root=data_root,
        vocab_file=vocab_file,
        class_file=class_file),
    val=dict(
        type=dataset_type,
        ann_file='test.txt',
        pipeline=test_pipeline,
        data_root=data_root,
        vocab_file=vocab_file,
        class_file=class_file),
    test=dict(
        type=dataset_type,
        ann_file='test.txt',
        pipeline=test_pipeline,
        data_root=data_root,
        vocab_file=vocab_file,
        class_file=class_file))

evaluation = dict(
    interval=1,
    metric='macro_f1',
    metric_options=dict(
        macro_f1=dict(
            ignores=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25])))

model = dict(
    type='SDMGR',
    backbone=dict(type='UNet', base_channels=16),
    bbox_head=dict(
        type='SDMGRHead', visual_dim=16, num_chars=92, num_classes=26),
    visual_modality=True,
    train_cfg=None,
    test_cfg=None)

optimizer = dict(type='Adam', weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1,
    warmup_ratio=1,
    step=[40, 50])
total_epochs = 60

checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(
        #     type='PaviLoggerHook',
        #     add_last_ckpt=True,
        #     interval=5,
        #     init_kwargs=dict(project='kie')),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

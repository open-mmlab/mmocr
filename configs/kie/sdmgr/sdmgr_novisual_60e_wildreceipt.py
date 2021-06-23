img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
max_scale, min_scale = 1024, 512

train_pipeline = [
    dict(type='LoadAnnotations'),
    dict(
        type='ResizeNoImg', img_scale=(max_scale, min_scale), keep_ratio=True),
    dict(type='KIEFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'relations', 'texts', 'gt_bboxes', 'gt_labels'],
        meta_keys=())
]
test_pipeline = [
    dict(type='LoadAnnotations'),
    dict(
        type='ResizeNoImg', img_scale=(max_scale, min_scale), keep_ratio=True),
    dict(type='KIEFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'relations', 'texts', 'gt_bboxes'],
        meta_keys=())
]

dataset_type = 'KIEDataset'
data_root = 'data/wildreceipt'

loader = dict(
    type='HardDiskLoader',
    repeat=1,
    parser=dict(
        type='LineJsonParser',
        keys=['file_name', 'height', 'width', 'annotations']))

train = dict(
    type=dataset_type,
    ann_file=f'{data_root}/train.txt',
    pipeline=train_pipeline,
    img_prefix=data_root,
    loader=loader,
    dict_file=f'{data_root}/dict.txt',
    test_mode=False)
test = dict(
    type=dataset_type,
    ann_file=f'{data_root}/test.txt',
    pipeline=test_pipeline,
    img_prefix=data_root,
    loader=loader,
    dict_file=f'{data_root}/dict.txt',
    test_mode=True)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=train,
    val=test,
    test=test)

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
    visual_modality=False,
    train_cfg=None,
    test_cfg=None,
    class_list=f'{data_root}/class_list.txt')

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
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

find_unused_parameters = True

_base_ = ['../../_base_/default_runtime.py']

model = dict(
    type='SDMGR',
    backbone=dict(type='UNet', base_channels=16),
    bbox_head=dict(
        type='SDMGRHead', visual_dim=16, num_chars=92, num_classes=4),
    visual_modality=False,
    train_cfg=None,
    test_cfg=None,
    class_list=None,
    openset=True)

optimizer = dict(type='Adam', weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1,
    warmup_ratio=1,
    step=[40, 50])
total_epochs = 60

train_pipeline = [
    dict(type='LoadAnnotations'),
    dict(type='ResizeNoImg', img_scale=(1024, 512), keep_ratio=True),
    dict(type='KIEFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'relations', 'texts', 'gt_bboxes', 'gt_labels'],
        meta_keys=('filename', 'ori_filename', 'ori_texts'))
]
test_pipeline = [
    dict(type='LoadAnnotations'),
    dict(type='ResizeNoImg', img_scale=(1024, 512), keep_ratio=True),
    dict(type='KIEFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'relations', 'texts', 'gt_bboxes'],
        meta_keys=('filename', 'ori_filename', 'ori_texts', 'ori_boxes',
                   'img_norm_cfg', 'ori_filename', 'img_shape'))
]

dataset_type = 'OpensetKIEDataset'
data_root = 'data/wildreceipt'

loader = dict(
    type='HardDiskLoader',
    repeat=1,
    parser=dict(
        type='LineJsonParser',
        keys=['file_name', 'height', 'width', 'annotations']))

train = dict(
    type=dataset_type,
    ann_file=f'{data_root}/openset_train.txt',
    pipeline=train_pipeline,
    img_prefix=data_root,
    link_type='one-to-many',
    loader=loader,
    dict_file=f'{data_root}/dict.txt',
    test_mode=False)
test = dict(
    type=dataset_type,
    ann_file=f'{data_root}/openset_test.txt',
    pipeline=test_pipeline,
    img_prefix=data_root,
    link_type='one-to-many',
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

evaluation = dict(interval=1, metric='openset_f1', metric_options=None)

find_unused_parameters = True

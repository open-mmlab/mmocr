_base_ = [
    '_base_spts_resnet50.py',
    '../_base_/datasets/icdar2015-spts.py',
    '../_base_/default_runtime.py',
]

load_from = 'work_dirs/spts_resnet50_150e_pretrain-spts/epoch_150.pth'

num_epochs = 350
lr = 0.00001

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='e2e_icdar/hmean',
        rule='greater',
        _delete_=True),
    logger=dict(type='LoggerHook', interval=10))

optim_wrapper = dict(
    type='OptimWrapper',
    accumulative_counts=2,
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.0001),
    paramwise_cfg=dict(custom_keys={
        'backbone': dict(lr_mult=0.1),
    }))

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=num_epochs, val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# dataset settings
icdar2015_textspotting_train = _base_.icdar2015_textspotting_train
icdar2015_textspotting_train.pipeline = _base_.train_pipeline
icdar2015_textspotting_test = _base_.icdar2015_textspotting_test
icdar2015_textspotting_test.pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    sampler=dict(type='RepeatAugSampler', shuffle=True, num_repeats=2),
    dataset=icdar2015_textspotting_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=icdar2015_textspotting_test)

test_dataloader = val_dataloader

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

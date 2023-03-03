_base_ = [
    '_base_spts_resnet50.py',
    '../_base_/datasets/icdar2013-spts.py',
    '../_base_/datasets/icdar2015-spts.py',
    '../_base_/datasets/totaltext-spts.py',
    '../_base_/datasets/syntext1-spts.py',
    '../_base_/datasets/syntext2-spts.py',
    '../_base_/datasets/mlt-spts.py',
    '../_base_/default_runtime.py',
]

num_epochs = 150
lr = 0.0005
min_lr = 0.00001

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.0001),
    paramwise_cfg=dict(custom_keys={
        'backbone': dict(lr_mult=0.1),
    }))
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=num_epochs, val_interval=5)
# learning policy
param_scheduler = [
    dict(type='LinearLR', end=5, start_factor=1 / 5, by_epoch=True),
    dict(
        type='LinearLR',
        begin=5,
        end=min(num_epochs,
                int((lr - min_lr) / (lr / num_epochs)) + 5),
        end_factor=min_lr / lr,
        by_epoch=True),
]

# dataset settings
train_list = [
    _base_.icdar2013_textspotting_train,
    _base_.icdar2015_textspotting_train,
    _base_.mlt_textspotting_train,
    _base_.totaltext_textspotting_train,
    _base_.syntext1_textspotting_train,
    _base_.syntext2_textspotting_train,
]

train_dataset = dict(
    type='ConcatDataset', datasets=train_list, pipeline=_base_.train_pipeline)

val_evaluator = None
test_evaluator = None

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    sampler=dict(type='BatchAugSampler', shuffle=True, num_repeats=2),
    dataset=train_dataset)

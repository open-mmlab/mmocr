_base_ = [
    '_base_psenet_resnet50_fpnf.py',
    '../_base_/datasets/icdar2015.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_600e.py',
]

# optimizer
optim_wrapper = dict(optimizer=dict(lr=1e-4))
train_cfg = dict(val_interval=40)
param_scheduler = [
    dict(type='MultiStepLR', milestones=[200, 400], end=600),
]

# dataset settings
icdar2015_textdet_train = _base_.icdar2015_textdet_train
icdar2015_textdet_test = _base_.icdar2015_textdet_test

# use quadrilaterals for icdar2015
model = dict(
    backbone=dict(style='pytorch'),
    det_head=dict(postprocessor=dict(text_repr_type='quad')))

# pipeline settings
icdar2015_textdet_train.pipeline = _base_.train_pipeline
icdar2015_textdet_test.pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=icdar2015_textdet_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=icdar2015_textdet_test)

test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=64 * 4)

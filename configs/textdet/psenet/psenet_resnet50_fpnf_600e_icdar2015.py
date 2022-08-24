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
ic15_det_train = _base_.ic15_det_train
ic15_det_test = _base_.ic15_det_test

# use quadrilaterals for icdar2015
model = dict(
    backbone=dict(style='pytorch'),
    det_head=dict(postprocessor=dict(text_repr_type='quad')))

# pipeline settings
ic15_det_train.pipeline = _base_.train_pipeline
ic15_det_test.pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=ic15_det_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=ic15_det_test)

test_dataloader = val_dataloader

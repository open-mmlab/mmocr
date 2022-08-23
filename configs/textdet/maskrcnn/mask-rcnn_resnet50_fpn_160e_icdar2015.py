_base_ = [
    '_base_mask-rcnn_resnet50_fpn.py',
    '../_base_/datasets/icdar2015.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_1200e.py',
]

# optimizer
optim_wrapper = dict(optimizer=dict(lr=0.08))
train_cfg = dict(max_epochs=160)
# learning policy
param_scheduler = [
    dict(type='LinearLR', end=500, start_factor=0.001, by_epoch=False),
    dict(type='MultiStepLR', milestones=[80, 128], end=160),
]

# dataset settings
ic15_det_train = _base_.ic15_det_train
ic15_det_test = _base_.ic15_det_test
ic15_det_train.pipeline = _base_.train_pipeline
ic15_det_test.pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=ic15_det_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=ic15_det_test)

test_dataloader = val_dataloader

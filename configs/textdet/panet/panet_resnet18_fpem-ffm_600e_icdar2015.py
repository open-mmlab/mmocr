_base_ = [
    '../_base_/datasets/icdar2015.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_600e.py',
    '_base_panet_resnet18_fpem-ffm.py',
]

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=20), )

# dataset settings
ic15_det_train = _base_.ic15_det_train
ic15_det_test = _base_.ic15_det_test
# pipeline settings
ic15_det_train.pipeline = _base_.train_pipeline
ic15_det_test.pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=ic15_det_train)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=ic15_det_test)
test_dataloader = val_dataloader

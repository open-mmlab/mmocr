_base_ = [
    '../_base_/datasets/icdar2015.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_600e.py',
    '_base_panet_resnet18_fpem-ffm.py',
]

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=20), )

# dataset settings
icdar2015_textdet_train = _base_.icdar2015_textdet_train
icdar2015_textdet_test = _base_.icdar2015_textdet_test
# pipeline settings
icdar2015_textdet_train.pipeline = _base_.train_pipeline
icdar2015_textdet_test.pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=icdar2015_textdet_train)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=icdar2015_textdet_test)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='HmeanIOUMetric', pred_score_thrs=dict(start=0.3, stop=1, step=0.05))
test_evaluator = val_evaluator

auto_scale_lr = dict(base_batch_size=64)

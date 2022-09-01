_base_ = [
    '../_base_/datasets/toy_data.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_base.py',
    '_base_nrtr_modality-transform.py',
]

# optimizer settings
train_cfg = dict(max_epochs=6)
# learning policy
param_scheduler = [
    dict(type='MultiStepLR', milestones=[3, 4], end=6),
]

# dataset settings
train_list = [_base_.toy_rec_train]
test_list = [_base_.toy_rec_test]

train_dataset = dict(
    type='ConcatDataset', datasets=train_list, pipeline=_base_.train_pipeline)
test_dataset = dict(
    type='ConcatDataset', datasets=test_list, pipeline=_base_.test_pipeline)

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)

test_dataloader = val_dataloader

val_evaluator = dict(dataset_prefixes=['Toy'])
test_evaluator = val_evaluator

_base_ = [
    '_base_master_resnet31.py',
    '../../_base_/recog_datasets/toy_data.py',
    '../../_base_/textrec_default_runtime.py',
    '../../_base_/schedules/schedule_adam_step_12e.py',
]

# dataset settings
train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}
file_client_args = dict(backend='disk')

train_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=train_list,
        pipeline=_base_.train_pipeline))

val_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=test_list,
        pipeline=_base_.test_pipeline))

test_dataloader = val_dataloader

val_evaluator = [
    dict(
        type='WordMetric', mode=['exact', 'ignore_case',
                                 'ignore_case_symbol']),
    dict(type='CharMetric')
]

test_evaluator = val_evaluator

_base_ = [
    '../../_base_/recog_datasets/toy_data.py',
    '../../_base_/textrec_default_runtime.py',
    '../../_base_/schedules/schedule_adam_step_6e.py',
    '_base_nrtr_modality-transform.py',
]

# dataset settings
train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

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

val_evaluator = [
    dict(
        type='WordMetric', mode=['exact', 'ignore_case',
                                 'ignore_case_symbol']),
    dict(type='CharMetric')
]

test_evaluator = val_evaluator

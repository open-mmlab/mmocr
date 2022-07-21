_base_ = [
    '_base_dbnet_r18_fpnc.py',
    '../../_base_/det_datasets/synthtext.py',
    '../../_base_/textdet_default_runtime.py',
    '../../_base_/schedules/schedule_sgd_100k_iters.py',
]

# dataset settings
train_list = [_base_.st_det_train]
test_list = [_base_.st_det_test]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=train_list,
        pipeline=_base_.train_pipeline_r18))

val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=test_list,
        pipeline=_base_.test_pipeline_1333_736))

test_dataloader = val_dataloader

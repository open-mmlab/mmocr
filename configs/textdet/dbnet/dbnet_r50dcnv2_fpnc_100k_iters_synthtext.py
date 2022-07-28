_base_ = [
    '_base_dbnet_r50dcnv2_fpnc.py',
    '../../_base_/textdet_default_runtime.py',
    '../../_base_/det_datasets/synthtext.py',
    '../../_base_/schedules/schedule_sgd_100k_iters.py',
]

# dataset settings
train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=train_list,
        pipeline=_base_.train_pipeline_r50dcnv2))
val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=test_list,
        pipeline=_base_.test_pipeline_4068_1024))

test_dataloader = val_dataloader

_base_ = [
    '_base_dbnet_r50dcnv2_fpnc.py',
    '../../_base_/det_datasets/icdar2015.py',
    '../../_base_/textdet_default_runtime.py',
    '../../_base_/schedules/schedule_sgd_1200e.py',
]

# dataset settings
train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

load_from = 'https://download.openmmlab.com/mmocr/textdet/dbnet/dbnet_r50dcnv2_fpnc_sbn_2e_synthtext_20210325-aa96e477.pth'  # noqa

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

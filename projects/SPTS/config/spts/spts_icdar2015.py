_base_ = [
    '_base_spts.py',
    '../_base_/datasets/icdar2015.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_500e.py',
]

# dataset settings
icdar2015_textspotting_train = _base_.icdar2015_textspotting_train
icdar2015_textspotting_train.pipeline = _base_.train_pipeline
icdar2015_textspotting_test = _base_.icdar2015_textspotting_test
icdar2015_textspotting_test.pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=icdar2015_textspotting_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=icdar2015_textspotting_test)

test_dataloader = val_dataloader

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

custom_imports = dict(imports='projects.SPTS.spts')

find_unused_parameters = True

_base_ = [
    '_base_abcnet-det_resnet50_fpn.py',
    '../_base_/datasets/icdar2015.py',
    '../_base_/default_runtime.py',
]

# dataset settings
icdar2015_textspotting_test = _base_.icdar2015_textspotting_test
icdar2015_textspotting_test.pipeline = _base_.test_pipeline

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=icdar2015_textspotting_test)

test_dataloader = val_dataloader

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

custom_imports = dict(
    imports=['projects.ABCNet.abcnet'], allow_failed_imports=False)

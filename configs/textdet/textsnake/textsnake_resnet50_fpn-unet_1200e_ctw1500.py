_base_ = [
    '_base_textsnake_resnet50_fpn-unet.py',
    '../../_base_/det_datasets/ctw1500.py',
    '../../_base_/textdet_default_runtime.py',
    '../../_base_/schedules/schedule_sgd_1200e.py',
]

# dataset settings
ctw_det_train = _base_.ctw_det_train
ctw_det_train.pipeline = _base_.train_pipeline
ctw_det_test = _base_.ctw_det_test
ctw_det_test.pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=ctw_det_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=ctw_det_test)

test_dataloader = val_dataloader

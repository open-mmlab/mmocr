_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/wildreceipt.py',
    '../_base_/schedules/schedule_adam_60e.py',
    '_base_sdmgr_novisual.py',
]

wildreceipt_train = _base_.wildreceipt_train
wildreceipt_train.pipeline = _base_.train_pipeline
wildreceipt_test = _base_.wildreceipt_test
wildreceipt_test.pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=4,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=wildreceipt_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=wildreceipt_test)
test_dataloader = val_dataloader
<<<<<<< HEAD
=======

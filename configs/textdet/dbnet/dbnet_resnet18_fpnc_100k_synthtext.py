_base_ = [
    '_base_dbnet_resnet18_fpnc.py',
    '../_base_/datasets/synthtext.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_100k.py',
]

# dataset settings
synthtext_textdet_train = _base_.synthtext_textdet_train
synthtext_textdet_train.pipeline = _base_.train_pipeline
synthtext_textdet_test = _base_.synthtext_textdet_test
synthtext_textdet_test.pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=synthtext_textdet_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=synthtext_textdet_test)

test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=16)

_base_ = [
    '_base_fcenet_resnet50_fpn.py',
    '../_base_/datasets/totaltext.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_base.py',
]

optim_wrapper = dict(optimizer=dict(lr=1e-3, weight_decay=5e-4))
train_cfg = dict(max_epochs=1500)
# learning policy
param_scheduler = [
    dict(type='PolyLR', power=0.9, eta_min=1e-7, end=1500),
]

# dataset settings
totaltext_textdet_train = _base_.totaltext_textdet_train
totaltext_textdet_test = _base_.totaltext_textdet_test
totaltext_textdet_train.pipeline = _base_.train_pipeline
totaltext_textdet_test.pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=totaltext_textdet_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=totaltext_textdet_test)

test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=8)

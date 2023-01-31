_base_ = [
    '_base_spts.py',
    '../_base_/datasets/ctw1500-spts.py',
    '../_base_/default_runtime.py',
]

num_epochs = 350
lr = 0.00001
min_lr = 0.00001

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.0001),
    paramwise_cfg=dict(custom_keys={
        'backbone': dict(lr_mult=0.1),
    }))

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=num_epochs, val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
# learning policy
param_scheduler = [
    dict(type='LinearLR', end=5, start_factor=1 / 5, by_epoch=True),
    dict(
        type='LinearLR',
        begin=5,
        end=min(num_epochs,
                int((lr - min_lr) / (lr / num_epochs)) + 5),
        end_factor=min_lr / lr,
        by_epoch=True),
]

# dataset settings
ctw1500_textspotting_train = _base_.ctw1500_textspotting_train
ctw1500_textspotting_train.pipeline = _base_.train_pipeline
ctw1500_textspotting_test = _base_.ctw1500_textspotting_test
ctw1500_textspotting_test.pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='RepeatAugSampler', shuffle=True, num_repeats=2),
    dataset=ctw1500_textspotting_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=ctw1500_textspotting_test)

test_dataloader = val_dataloader

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

custom_imports = dict(imports='spts')

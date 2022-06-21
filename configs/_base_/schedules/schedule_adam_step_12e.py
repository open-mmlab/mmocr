# optimizer

optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='Adam', lr=4e-4))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
# learning policy
param_scheduler = [
    dict(type='LinearLR', end=100, by_epoch=False),
    dict(type='MultiStepLR', milestones=[11], end=12),
]

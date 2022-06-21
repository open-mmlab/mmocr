# optimizer
optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='Adam', lr=1e-4))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=600, val_interval=40)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
# learning policy
param_scheduler = [
    dict(type='MultiStepLR', milestones=[200, 400], end=600),
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper', optimizer=dict(type='Adam', weight_decay=0.0001))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=60, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
# learning rate
param_scheduler = [
    dict(type='MultiStepLR', milestones=[40, 50], end=60),
]

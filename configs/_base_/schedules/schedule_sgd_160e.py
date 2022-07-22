# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.08, momentum=0.9, weight_decay=0.0001))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=160, val_interval=20)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
# learning policy
param_scheduler = [
    dict(type='LinearLR', end=500, start_factor=0.001, by_epoch=False),
    dict(type='MultiStepLR', milestones=[80, 128], end=160),
]

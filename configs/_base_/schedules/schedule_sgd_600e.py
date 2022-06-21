optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=1e-3, momentum=0.99, weight_decay=5e-4))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=600, val_interval=50)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
# learning policy
param_scheduler = [
    dict(type='MultiStepLR', milestones=[200, 400], end=600),
]

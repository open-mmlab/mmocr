# Note: This schedule config serves as a base config for other schedules.
# Users would have to at least fill in "max_epochs" and "val_interval"
# in order to use this config in their experiments.

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.007, momentum=0.9, weight_decay=0.0001))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=None, val_interval=20)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
# learning policy
param_scheduler = [
    dict(type='ConstantLR', factor=1.0),
]

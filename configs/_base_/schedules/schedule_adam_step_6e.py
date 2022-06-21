_base_ = 'schedule_adam_step_5e.py'

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=6, val_interval=1)

# learning policy
param_scheduler = [
    dict(type='MultiStepLR', milestones=[3, 4], end=6),
]

_base_ = 'schedule_adam_step_5e.py'

train_cfg = dict(by_epoch=True, max_epochs=6)

# learning policy
param_scheduler = [
    dict(type='MultiStepLR', milestones=[3, 4], end=6),
]

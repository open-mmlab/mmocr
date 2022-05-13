# optimizer
optimizer = dict(type='Adam', lr=4e-4)

train_cfg = dict(by_epoch=True, max_epochs=12)
val_cfg = dict(interval=1)
test_cfg = dict()

# learning policy
param_scheduler = [
    dict(type='LinearLR', end=100, by_epoch=False),
    dict(type='MultiStepLR', milestones=[11], end=12),
]

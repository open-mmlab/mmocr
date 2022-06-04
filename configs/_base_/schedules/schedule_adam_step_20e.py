# optimizer
optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='Adam', lr=1e-4))
train_cfg = dict(by_epoch=True, max_epochs=20)
val_cfg = dict(interval=1)
test_cfg = dict()

# learning policy
param_scheduler = [
    dict(type='LinearLR', end=1, start_factor=0.001),
    dict(type='MultiStepLR', milestones=[16, 18], end=20),
]

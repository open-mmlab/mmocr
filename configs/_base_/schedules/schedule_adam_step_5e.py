# optimizer
optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='Adam', lr=1e-3))
train_cfg = dict(by_epoch=True, max_epochs=5)
val_cfg = dict(interval=1)
test_cfg = dict()

# learning policy
param_scheduler = [
    dict(type='MultiStepLR', milestones=[3, 4], end=5),
]

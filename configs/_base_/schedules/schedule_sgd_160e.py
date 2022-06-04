# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.08, momentum=0.9, weight_decay=0.0001))
train_cfg = dict(by_epoch=True, max_epochs=160)
val_cfg = dict(interval=20)
test_cfg = dict()

# learning policy
param_scheduler = [
    dict(type='LinearLR', end=500, start_factor=0.001, by_epoch=False),
    dict(type='MultiStepLR', milestones=[80, 128], end=160),
]
